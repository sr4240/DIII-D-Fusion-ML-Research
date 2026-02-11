#!/usr/bin/env python3
"""
Fixed CNN for Plasma Four-State Classification
Corrects data shape and convolution axis issues from original implementation
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
                focal = alpha_t * focal
            else:
                focal = self.alpha * focal
        return focal.mean()


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time series"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, hidden_dim)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # Apply attention and sum over time dimension
        return torch.sum(attention_weights * x, dim=1)


class MultiScaleConv1d(nn.Module):
    """Multi-scale 1D convolution to capture patterns at different time scales"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Different kernel sizes for different time scales
        # Make sure outputs sum to out_channels
        ch1 = out_channels // 3
        ch2 = out_channels // 3
        ch3 = out_channels - ch1 - ch2  # Ensure exact match

        self.conv_short = nn.Conv1d(in_channels, ch1, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(in_channels, ch2, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(in_channels, ch3, kernel_size=15, padding=7)

        # Calculate actual concatenated channels
        self.concat_channels = ch1 + ch2 + ch3

        # Combine features
        self.combine = nn.Conv1d(self.concat_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Extract features at multiple scales
        short = F.relu(self.conv_short(x))
        medium = F.relu(self.conv_medium(x))
        long = F.relu(self.conv_long(x))

        # Concatenate along channel dimension
        multi_scale = torch.cat([short, medium, long], dim=1)

        # Combine and normalize
        out = self.combine(multi_scale)
        out = self.bn(out)
        return F.relu(out)


class FixedPlasmaDataset(Dataset):
    """Fixed dataset that returns data in correct shape for Conv1d"""
    def __init__(self, features, labels, shots, window_size=150):
        self.features = features.astype(np.float32)
        self.labels = labels
        self.shots = shots
        self.window_size = window_size
        self.windows, self.window_labels = self._create_windows()

        # Map to 0-indexed classes
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        self.window_labels = np.array([label_mapping[int(label)] for label in self.window_labels])

        print(f"Created {len(self.windows)} windows")
        print(f"Window shape: {self.windows[0].shape if len(self.windows) > 0 else 'N/A'}")
        print(f"Label distribution: {Counter(self.window_labels)}")

    def _create_windows(self):
        windows, window_labels = [], []
        half = self.window_size // 2

        for shot_id in np.unique(self.shots):
            shot_indices = np.where(self.shots == shot_id)[0]
            if len(shot_indices) < self.window_size:
                continue

            start_idx, end_idx = shot_indices[0], shot_indices[-1]

            for i in range(start_idx, end_idx - self.window_size + 2):
                if (i + self.window_size - 1) > end_idx:
                    break

                window_slice = slice(i, i + self.window_size)
                window_data = self.features[window_slice]  # Shape: (window_size, n_features)
                window_labels_slice = self.labels[window_slice]
                center_label = window_labels_slice[half]

                # Store window WITHOUT transposing
                windows.append(window_data)
                window_labels.append(center_label)

        return np.array(windows, dtype=np.float32), np.array(window_labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # FIXED: Return correct shape (n_features, window_size) for Conv1d
        # Original had .T which incorrectly transposed the data
        window = self.windows[idx]  # Shape: (window_size, n_features)

        # Transpose to (n_features, window_size) for Conv1d
        # This way Conv1d convolves ACROSS TIME for each feature
        window_tensor = torch.FloatTensor(window.T)  # Now shape: (n_features, window_size)
        label_tensor = torch.LongTensor([self.window_labels[idx]]).squeeze()

        return window_tensor, label_tensor


class FixedPlasmaWindowDataset(Dataset):
    """Fixed dataset for pre-created windows"""
    def __init__(self, windows, labels):
        self.windows = windows.astype(np.float32)
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # FIXED: Correct shape transformation
        # windows[idx] shape: (window_size, n_features)
        # Need: (n_features, window_size) for Conv1d
        window_tensor = torch.FloatTensor(self.windows[idx].T)
        label_tensor = torch.LongTensor([self.labels[idx]]).squeeze()
        return window_tensor, label_tensor


class FixedPlasmaCNN(nn.Module):
    """
    Fixed CNN architecture for plasma state classification
    Key fixes:
    1. Expects input shape (batch, n_features, window_size)
    2. Convolves across time dimension properly
    3. Uses multi-scale convolutions for different temporal patterns
    4. Includes proper batch normalization and dropout
    """

    def __init__(self, n_features, n_classes=4, window_size=150):
        super().__init__()

        # Store dimensions
        self.n_features = n_features
        self.n_classes = n_classes
        self.window_size = window_size

        # Multi-scale temporal feature extraction
        self.conv1 = MultiScaleConv1d(n_features, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)

        # Second conv layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )

        # Third conv layer
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        # Calculate size after convolutions
        # window_size=150: after 3 pooling layers of 2: 150 -> 75 -> 37 -> 18
        conv_output_size = window_size // 8  # Three MaxPool1d(2) layers

        # Bidirectional LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Attention mechanism
        self.attention = TemporalAttention(128)  # 64*2 for bidirectional

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {total_params:,} parameters")
        print(f"Input shape expected: (batch, {n_features}, {window_size})")

    def forward(self, x):
        # Input shape: (batch, n_features, window_size)
        # This is CORRECT for Conv1d

        # Multi-scale temporal convolution
        x = self.conv1(x)  # (batch, 32, window_size)
        x = self.pool1(x)  # (batch, 32, window_size//2)
        x = self.dropout1(x)

        # Further convolutions
        x = self.conv2(x)  # (batch, 64, window_size//4)
        x = self.conv3(x)  # (batch, 128, window_size//8)

        # Prepare for LSTM: (batch, sequence_length, features)
        x = x.permute(0, 2, 1)  # (batch, window_size//8, 128)

        # LSTM processing
        x, _ = self.lstm(x)  # (batch, window_size//8, 128)

        # Apply attention
        x = self.attention(x)  # (batch, 128)

        # Classification
        x = self.classifier(x)  # (batch, n_classes)

        return x


class CausalPlasmaCNN(nn.Module):
    """
    Causal CNN that only uses past information for center point prediction
    Better for real-time applications
    """

    def __init__(self, n_features, n_classes=4, window_size=150):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.window_size = window_size

        # Causal padding for convolutions
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((6, 0), 0),  # Pad only left side
            nn.Conv1d(n_features, 32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((4, 0), 0),  # Pad only left side
            nn.Conv1d(32, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0),  # Pad only left side
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

        print(f"Causal CNN initialized for real-time prediction")

    def forward(self, x):
        # Only use past and current information
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global pooling
        x = self.global_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)

        # Classify
        x = self.classifier(x)

        return x


def load_and_prepare_data():
    """Load and prepare plasma data"""
    print("Loading and preparing data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot if needed
    df = df[df['shot'] != 191675].copy()

    # Feature selection - important plasma parameters
    important_features = ['iln3iamp', 'betan', 'density', 'n_eped', 'li', 'tritop','fs04_max_smoothed'
    ]

    # Select available features
    selected_features = [f for f in important_features if f in df.columns]
    print(f"Selected {len(selected_features)} features: {selected_features[:5]}...")

    # Sort by shot and time for proper windowing
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Filter out state 0 (undefined)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()

    # Prepare features and labels
    X = df_filtered[selected_features].values
    y = df_filtered['state'].values
    shots = df_filtered['shot'].values

    # Handle missing values
    print(f"Original data shape: {X.shape}")
    print(f"NaN values in features: {np.isnan(X).sum()}")

    # Remove rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    shots = shots[valid_mask]

    print(f"After removing NaN: {X.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check for infinite values
    inf_mask = np.isinf(X_scaled)
    if inf_mask.any():
        print(f"Warning: {inf_mask.sum()} infinite values found, clipping...")
        X_scaled = np.clip(X_scaled, -10, 10)

    return X_scaled, y, shots, selected_features, scaler


def create_windows_and_split(X, y, shots, window_size=150):
    """Create windows and split into train/val/test sets"""
    print(f"Creating windows of size {window_size}...")

    all_windows, all_labels = [], []
    half = window_size // 2

    unique_shots = np.unique(shots)
    print(f"Processing {len(unique_shots)} shots...")

    for shot_id in unique_shots:
        shot_indices = np.where(shots == shot_id)[0]

        if len(shot_indices) < window_size:
            continue

        start_idx, end_idx = shot_indices[0], shot_indices[-1]

        # Sliding window with stride
        stride = window_size // 4  # 75% overlap

        for i in range(start_idx, end_idx - window_size + 2, stride):
            if (i + window_size - 1) > end_idx:
                break

            window_slice = slice(i, i + window_size)
            window_data = X[window_slice]
            window_labels = y[window_slice]
            center_label = window_labels[half]

            # Verify window belongs to same shot
            window_shots = shots[window_slice]
            if len(np.unique(window_shots)) == 1:
                # Check for data quality
                if not np.isnan(window_data).any() and not np.isinf(window_data).any():
                    all_windows.append(window_data)
                    all_labels.append(center_label)

    all_windows = np.array(all_windows, dtype=np.float32)
    all_labels = np.array(all_labels)

    print(f"Created {len(all_windows)} windows")
    print(f"Window shape: {all_windows[0].shape}")
    print(f"Label distribution: {Counter(all_labels)}")

    # Map labels to 0-indexed
    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    all_labels_mapped = np.array([label_mapping[int(label)] for label in all_labels])

    # Random split
    np.random.seed(42)
    indices = np.random.permutation(len(all_windows))

    train_size = int(0.7 * len(all_windows))
    val_size = int(0.15 * len(all_windows))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Split data
    train_windows = all_windows[train_indices]
    train_labels = all_labels_mapped[train_indices]

    val_windows = all_windows[val_indices]
    val_labels = all_labels_mapped[val_indices]

    test_windows = all_windows[test_indices]
    test_labels = all_labels_mapped[test_indices]

    print(f"Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")

    return train_windows, train_labels, val_windows, val_labels, test_windows, test_labels


def create_data_loaders(X, y, shots, window_size=150, batch_size=64):
    """Create data loaders with proper handling"""

    # Create windows and split
    train_windows, train_labels, val_windows, val_labels, test_windows, test_labels = \
        create_windows_and_split(X, y, shots, window_size)

    # Create datasets
    train_dataset = FixedPlasmaWindowDataset(train_windows, train_labels)
    val_dataset = FixedPlasmaWindowDataset(val_windows, val_labels)
    test_dataset = FixedPlasmaWindowDataset(test_windows, test_labels)

    # Class balancing for training
    class_counts = Counter(train_labels)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_and_evaluate(train_loader, val_loader, test_loader, n_features, use_causal=False):
    """Train and evaluate the fixed CNN model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Choose model type
    if use_causal:
        print("Using Causal CNN (only past information)")
        model = CausalPlasmaCNN(n_features=n_features, n_classes=4).to(device)
    else:
        print("Using Standard Fixed CNN")
        model = FixedPlasmaCNN(n_features=n_features, n_classes=4).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Loss with class weights
    all_train_labels = train_loader.dataset.labels
    classes_present = np.unique(all_train_labels)
    class_weights = compute_class_weight('balanced', classes=classes_present, y=all_train_labels)

    # Create weight tensor for all 4 classes
    full_weights = np.ones(4, dtype=np.float32)
    for cls, weight in zip(classes_present, class_weights):
        full_weights[int(cls)] = weight

    class_weights_tensor = torch.tensor(full_weights).to(device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)

    # Training parameters
    n_epochs = 15
    patience = 5
    best_val_accuracy = 0.0
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 50)

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels_list = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(batch_labels.cpu().numpy())

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = accuracy_score(val_labels_list, val_predictions)

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'fixed_plasma_cnn_best.pth')
            patience_counter = 0
            print(f"  ✓ New best model saved (Val Acc: {val_accuracy:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load('fixed_plasma_cnn_best.pth'))
    model.eval()

    # Test evaluation
    test_predictions = []
    test_labels_list = []
    test_probabilities = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)

            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            test_predictions.extend(predicted.cpu().numpy())
            test_labels_list.extend(batch_labels.numpy())
            test_probabilities.extend(probabilities.cpu().numpy())

    # Convert to arrays
    test_predictions = np.array(test_predictions)
    test_labels_list = np.array(test_labels_list)
    test_probabilities = np.array(test_probabilities)

    # Print results
    print("\n" + "=" * 50)
    print("Test Set Results")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(test_labels_list, test_predictions,
                              target_names=['L-Mode', 'Dithering', 'H-Mode ELM-free', 'ELMing']))

    # Calculate and print ROC AUC for each class
    print("\nROC AUC Scores:")
    for i, class_name in enumerate(['L-Mode', 'Dithering', 'H-Mode ELM-free', 'ELMing']):
        class_labels = (test_labels_list == i).astype(int)
        class_probs = test_probabilities[:, i]

        if len(np.unique(class_labels)) > 1:
            auc_score = roc_auc_score(class_labels, class_probs)
            print(f"  {class_name}: {auc_score:.4f}")

    # Calculate test accuracy
    test_accuracy = accuracy_score(test_labels_list, test_predictions)
    print(f"\nOverall Test Accuracy: {test_accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels_list, test_predictions, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['L-Mode', 'Dithering', 'H-Mode', 'ELMing'],
                yticklabels=['L-Mode', 'Dithering', 'H-Mode', 'ELMing'])
    plt.title('Fixed CNN - Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('fixed_cnn_confusion_matrix.png', dpi=300)
    plt.close()  # Close figure to avoid display issues

    return model, test_accuracy


def main():
    """Main execution function"""
    print("=" * 60)
    print("Fixed CNN for Plasma Four-State Classification")
    print("=" * 60)
    print("\nKey fixes applied:")
    print("1. ✓ Correct data shape: (batch, n_features, window_size)")
    print("2. ✓ Convolutions across time dimension")
    print("3. ✓ Multi-scale temporal feature extraction")
    print("4. ✓ Proper batch normalization and dropout")
    print("5. ✓ Option for causal convolutions\n")

    # Load and prepare data
    X, y, shots, selected_features, scaler = load_and_prepare_data()

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y, shots, window_size=150, batch_size=64
    )

    # Train and evaluate
    model, test_acc = train_and_evaluate(
        train_loader, val_loader, test_loader,
        n_features=len(selected_features),
        use_causal=False  # Set to True for causal CNN
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()