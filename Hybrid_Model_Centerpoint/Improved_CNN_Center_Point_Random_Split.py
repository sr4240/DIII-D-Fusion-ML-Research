import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
    def __init__(self, alpha=0, gamma=2.0):
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

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.fc_out(attention_output)

        return output

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x

class ImprovedPlasmaClassifier(nn.Module):
    def __init__(self, n_features, n_classes=4):
        super().__init__()

        # Enhanced CNN with residual blocks
        self.feature_conv = nn.Sequential(
            # First block
            nn.Conv1d(n_features, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Residual blocks
            ResidualBlock(32, 64, kernel_size=5, stride=2),
            ResidualBlock(64, 128, kernel_size=5, stride=2),
            ResidualBlock(128, 256, kernel_size=3, stride=2),
        )

        # Deeper BiLSTM with more capacity
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim=256, num_heads=8)

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 256),  # *3 for attention + avg pool + max pool
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        # x shape: (batch_size, n_features, window_size)
        x = self.feature_conv(x)  # (batch_size, 256, ~19)

        # LSTM processing
        x_lstm = x.permute(0, 2, 1)  # (batch_size, seq_len, 256)
        x_lstm, _ = self.lstm(x_lstm)  # (batch_size, seq_len, 256)

        # Multi-head attention
        x_attn = self.attention(x_lstm)  # (batch_size, seq_len, 256)
        x_attn_pooled = torch.mean(x_attn, dim=1)  # (batch_size, 256)

        # Global pooling on conv features
        x_avg = self.global_avg_pool(x).squeeze(-1)  # (batch_size, 256)
        x_max = self.global_max_pool(x).squeeze(-1)  # (batch_size, 256)

        # Concatenate all features
        x_combined = torch.cat([x_attn_pooled, x_avg, x_max], dim=1)  # (batch_size, 768)

        # Classification
        x = self.classifier(x_combined)
        return x

class PlasmaWindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows.astype(np.float32)
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]).T, torch.LongTensor([self.labels[idx]]).squeeze()

def load_and_prepare_data():
    print("Loading and preparing data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')
    df = df[df['shot'] != 191675].copy()

    # Feature selection
    columns_to_remove = ['ELMPMARKER', 'zeff', 'rotation_edge', 'rotation_core']
    important_features = ['n_eped', 'li', 'q95', 'bt0', 'kappa', 'tribot', 'tritop', 'dR_sep', 'fs04_max_smoothed', 'iln3iamp', 'fs04_max_avg', 'fs_sum', 'fs_up_sum']
    selected_features = [f for f in important_features if f in df.columns]

    # Data preparation
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()

    # Prepare features and labels
    X = df_filtered[selected_features].values
    y = df_filtered['state'].values
    shots = df_filtered['shot'].values

    # Check for NaN values and handle them
    print(f"Original data shape: {X.shape}")
    print(f"NaN values in features: {np.isnan(X).sum()}")
    print(f"NaN values in labels: {np.isnan(y).sum()}")

    # Remove rows with NaN values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    shots = shots[valid_mask]

    print(f"After removing NaN: {X.shape}")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check for infinite values
    print(f"Infinite values in scaled features: {np.isinf(X_scaled).sum()}")

    return X_scaled, y, shots, selected_features, scaler

def create_windows_and_split(X, y, shots, window_size=150):
    """
    Create windows ensuring all points in a window belong to the same shot,
    then split windows randomly into train/val/test sets.
    """
    print("Creating windows with shot-based constraints...")

    # Create all windows first
    all_windows, all_labels = [], []
    half = window_size // 2

    for shot_id in np.unique(shots):
        shot_indices = np.where(shots == shot_id)[0]
        if len(shot_indices) < window_size:
            continue

        start_idx, end_idx = shot_indices[0], shot_indices[-1]
        for i in range(start_idx, end_idx - window_size + 2):
            if (i + window_size - 1) > end_idx:
                break

            window_slice = slice(i, i + window_size)
            window_data = X[window_slice]
            window_labels_slice = y[window_slice]
            center_label = window_labels_slice[half]

            # Verify all points in window belong to the same shot
            window_shots = shots[window_slice]
            if len(np.unique(window_shots)) == 1:  # All points belong to same shot
                # Check for NaN or infinite values in window
                if not np.isnan(window_data).any() and not np.isinf(window_data).any():
                    all_windows.append(window_data)
                    all_labels.append(center_label)

    all_windows = np.array(all_windows, dtype=np.float32)
    all_labels = np.array(all_labels)

    print(f"Created {len(all_windows)} windows | Label distribution: {Counter(all_labels)}")

    # Four-state classification mapping
    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    all_labels_mapped = np.array([label_mapping[int(label)] for label in all_labels])

    # Random split of windows
    np.random.seed(42)
    indices = np.random.permutation(len(all_windows))

    train_count = int(0.75 * len(all_windows))
    val_count = int(0.15 * len(all_windows))

    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]

    # Split windows
    train_windows = all_windows[train_indices]
    train_labels = all_labels_mapped[train_indices]

    val_windows = all_windows[val_indices]
    val_labels = all_labels_mapped[val_indices]

    test_windows = all_windows[test_indices]
    test_labels = all_labels_mapped[test_indices]

    return train_windows, train_labels, val_windows, val_labels, test_windows, test_labels

def create_data_loaders(X, y, shots, window_size=150):
    # Create windows and split them
    train_windows, train_labels, val_windows, val_labels, test_windows, test_labels = create_windows_and_split(X, y, shots, window_size)

    # Create datasets
    train_dataset = PlasmaWindowDataset(train_windows, train_labels)
    val_dataset = PlasmaWindowDataset(val_windows, val_labels)
    test_dataset = PlasmaWindowDataset(test_windows, test_labels)

    # Weighted sampler for training
    class_counts = Counter(train_dataset.labels)
    class_weights_list = [1.0 / max(1, class_counts.get(c, 0)) for c in range(4)]
    sample_weights = np.array([class_weights_list[label] for label in train_dataset.labels], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def train_and_evaluate(train_loader, val_loader, test_loader, n_features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model setup with improved architecture
    model = ImprovedPlasmaClassifier(n_features=n_features, n_classes=4).to(device)

    # Improved optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Class weights
    all_train_labels = train_loader.dataset.labels
    classes_present = np.unique(all_train_labels)
    class_weights = compute_class_weight('balanced', classes=classes_present, y=all_train_labels)
    # Ensure weights align to class indices [0,1,2,3]
    full_weights = np.zeros(4, dtype=np.float32)
    for cls, w in zip(classes_present, class_weights):
        full_weights[int(cls)] = w
    class_weights_t = torch.tensor(full_weights, dtype=torch.float).to(device)
    criterion = FocalLoss(alpha=class_weights_t, gamma=2.0)

    # Training
    n_epochs, patience = 120, 15
    best_val_accuracy, patience_counter = 0.0, 0

    # Lists to store training history
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_predictions, train_labels_list = [], []
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Check for NaN in input
            if torch.isnan(batch_features).any():
                print(f"NaN detected in batch features at epoch {epoch}")
                continue

            optimizer.zero_grad()
            outputs = model(batch_features)

            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                print(f"NaN detected in model outputs at epoch {epoch}")
                continue

            loss = criterion(outputs, batch_labels)

            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"NaN detected in loss at epoch {epoch}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

            # Store predictions for accuracy calculation
            _, preds = torch.max(outputs, 1)
            train_predictions.extend(preds.cpu().numpy())
            train_labels_list.extend(batch_labels.cpu().numpy())

        # Validation
        model.eval()
        val_loss, val_predictions, val_labels = 0.0, [], []
        val_batches = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
                val_batches += 1

        if train_batches > 0 and val_batches > 0:
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            train_accuracy = accuracy_score(train_labels_list, train_predictions)
            val_accuracy = accuracy_score(val_labels, val_predictions)

            # Store metrics for plotting
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

            # Learning rate scheduling
            scheduler.step(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best_improved_plasma_classifier.pth')
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            print(f"Epoch {epoch+1}: No valid batches processed")
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Plot training curves
    if len(train_losses) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves_improved.png', dpi=800, bbox_inches='tight')
        plt.show()

        print(f"\nTraining Summary:")
        print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
        print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
        print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_improved_plasma_classifier.pth'))
    model.eval()

    # Threshold tuning on validation set
    print("\nPerforming threshold tuning...")
    val_probs, val_true = [], []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            val_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            val_true.extend(batch_labels.numpy())

    val_probs = np.array(val_probs)
    val_true = np.array(val_true)

    # Try different thresholds for each class
    best_thresholds = [0.5, 0.5, 0.5, 0.5]  # Default thresholds
    best_f1_scores = [0, 0, 0, 0]

    for class_idx in range(4):
        class_probs = val_probs[:, class_idx]
        class_labels = (val_true == class_idx).astype(int)

        # Try different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (class_probs >= threshold).astype(int)
            if len(np.unique(preds)) > 1:  # Only if both classes are present
                f1 = f1_score(class_labels, preds, average='binary')
                if f1 > best_f1_scores[class_idx]:
                    best_f1_scores[class_idx] = f1
                    best_thresholds[class_idx] = threshold

    print(f"Best thresholds: {best_thresholds}")
    print(f"Best F1 scores: {best_f1_scores}")

    # Test evaluation
    all_predictions, all_labels, all_probabilities = [], [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Results
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions,
                               target_names=['Suppressed', 'Dithering', 'Mitigated', 'ELMing'],
                               digits=5))

    # Multi-class ROC AUC (one-vs-rest)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    # Calculate ROC AUC for each class
    roc_auc_scores = []
    for i in range(4):
        class_labels = (all_labels == i).astype(int)
        class_probs = all_probabilities[:, i]
        if len(np.unique(class_labels)) > 1:  # Only if both classes are present
            auc_score = roc_auc_score(class_labels, class_probs)
            roc_auc_scores.append(auc_score)
            print(f"ROC AUC for class {i} ({['Suppressed', 'Dithering', 'Mitigated', 'ELMing'][i]}): {auc_score:.6f}")

    if roc_auc_scores:
        mean_auc = np.mean(roc_auc_scores)
        print(f"Mean ROC AUC Score: {mean_auc:.6f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Suppressed', 'Dithering', 'Mitigated', 'ELMing'],
                yticklabels=['Suppressed', 'Dithering', 'Mitigated', 'ELMing'])
    plt.title('Normalized Confusion Matrix (Improved Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot ROC curves for each class
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    class_names = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']

    for i in range(4):
        class_labels = (all_labels == i).astype(int)
        class_probs = all_probabilities[:, i]
        if len(np.unique(class_labels)) > 1:
            fpr, tpr, _ = roc_curve(class_labels, class_probs)
            auc_score = roc_auc_score(class_labels, class_probs)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_names[i]} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Improved Model)')
    plt.legend()
    plt.savefig('roc_curves_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Improved Plasma Four-State Classification ===")
    X, y, shots, selected_features, scaler = load_and_prepare_data()
    train_loader, val_loader, test_loader = create_data_loaders(X, y, shots)
    train_and_evaluate(train_loader, val_loader, test_loader, n_features=len(selected_features))
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()