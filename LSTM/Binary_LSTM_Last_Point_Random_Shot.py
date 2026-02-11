import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class LSTMFirstNN(nn.Module):
    """
    A hybrid model with LSTM processing FIRST (for temporal patterns)
    followed by NN layers (for feature transformation).
    Uses 150 datapoints BEFORE the classification point.
    Unidirectional LSTM only (not bidirectional).
    
    BINARY CLASSIFICATION:
    - Class 0: Suppressed (original state 1)
    - Class 1: Dithering + Mitigated + ELMing (original states 2, 3, 4)
    """
    def __init__(self, n_features, n_classes=2, lstm_hidden=128, nn_hidden_sizes=[256, 128]):
        super(LSTMFirstNN, self).__init__()

        # LSTM processes the raw temporal data FIRST
        # Unidirectional for last-point prediction
        self.lstm = nn.LSTM(
            input_size=n_features,  # Direct input of raw features
            hidden_size=lstm_hidden,
            num_layers=2,  # Deeper LSTM for better temporal learning
            batch_first=True,
            bidirectional=False,  # Unidirectional for forward-in-time prediction
            dropout=0.2
        )

        # After LSTM, we have temporal features
        lstm_output_size = lstm_hidden  # Unidirectional

        # NN layers process the LSTM output
        nn_layers = []
        input_dim = lstm_output_size

        for hidden_size in nn_hidden_sizes:
            nn_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.25)
            ])
            input_dim = hidden_size

        self.nn_layers = nn.Sequential(*nn_layers)

        # Feature aggregation from sequence
        self.attention_weights = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Softmax(dim=1)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim + lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

        # Print detailed model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count parameters by component
        lstm_params = sum(p.numel() for name, p in self.named_parameters() if 'lstm' in name)
        nn_params = sum(p.numel() for name, p in self.named_parameters() if 'nn_layers' in name)
        attention_params = sum(p.numel() for name, p in self.named_parameters() if 'attention' in name)
        classifier_params = sum(p.numel() for name, p in self.named_parameters() if 'classifier' in name)

        print(f"\n{'='*60}")
        print(f"LSTM-First-NN Model Parameter Count (BINARY):")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nParameters by component:")
        print(f"  - LSTM layers: {lstm_params:,} ({lstm_params/total_params*100:.1f}%)")
        print(f"  - NN layers: {nn_params:,} ({nn_params/total_params*100:.1f}%)")
        print(f"  - Attention: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
        print(f"  - Classifier: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Architecture: LSTM (unidirectional) → NN → Binary Classifier")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        batch_size, n_features, seq_len = x.shape

        # Transpose for LSTM: (batch_size, sequence_length, n_features)
        x = x.transpose(1, 2)

        # STEP 1: LSTM processes the temporal sequence
        lstm_output, (hidden, cell) = self.lstm(x)
        # lstm_output shape: (batch_size, seq_len, lstm_hidden)

        # STEP 2: Apply attention to aggregate temporal information
        attention = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        attended_features = torch.sum(lstm_output * attention, dim=1)  # (batch_size, lstm_hidden)

        # STEP 3: Process the final LSTM hidden state through NN
        # Take the last hidden state (for last-point prediction)
        final_hidden = lstm_output[:, -1, :]  # (batch_size, lstm_hidden)

        # Process through NN layers
        nn_features = self.nn_layers(final_hidden)  # (batch_size, nn_hidden[-1])

        # STEP 4: Combine attended features with NN features
        combined = torch.cat([nn_features, attended_features], dim=1)

        # STEP 5: Final classification
        output = self.classifier(combined)

        return output

class PlasmaDataset(Dataset):
    """Dataset class for plasma data windows"""
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Transpose to get (n_features, sequence_length) format
        return self.windows[idx].T, self.labels[idx]

def load_and_prepare_data():
    """Load and preprocess the plasma data"""
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    # Select only the specified 7 features
    important_features = ['iln3iamp', 'betan', 'density', 'li',
                         'tritop', 'fs04_past_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    print(f"Using {len(selected_features)} features: {selected_features}")

    # Sort by shot and time
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Filter out state 0
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()

    # Extract features and labels
    X = df_filtered[selected_features].values
    y = df_filtered['state'].values
    shots = df_filtered['shot'].values

    # Remove NaN values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    shots = shots[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Original label distribution: {Counter(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, shots, selected_features, scaler

def create_windows_with_random_shot_split(X, y, shots, window_size=150):
    """Create windows and perform random split BY SHOT - using LAST point label
    
    BINARY CLASSIFICATION:
    - Class 0: Suppressed (original state 1)
    - Class 1: Dithering + Mitigated + ELMing (original states 2, 3, 4)
    
    This function splits data by shot number, ensuring all windows from the same
    shot end up in the same split (train, val, or test).
    """
    print(f"Creating windows of size {window_size} (predicting last point)...")
    print("Splitting by SHOT NUMBER (not individual data points)")
    print("\nBINARY CLASSIFICATION:")
    print("  Class 0: Suppressed (state 1)")
    print("  Class 1: Dithering + Mitigated + ELMing (states 2, 3, 4)")

    # Get unique shots
    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"\nTotal unique shots: {n_shots}")

    # Randomly shuffle shots
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)

    # Split shots into train/val/test (70/15/15)
    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)

    train_shots = set(shuffled_shots[:train_size])
    val_shots = set(shuffled_shots[train_size:train_size + val_size])
    test_shots = set(shuffled_shots[train_size + val_size:])

    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}")

    # Create windows for each split
    train_windows, train_labels = [], []
    val_windows, val_labels = [], []
    test_windows, test_labels = [], []

    # BINARY Label mapping:
    # State 1 (Suppressed) -> 0
    # States 2, 3, 4 (Dithering, Mitigated, ELMing) -> 1
    label_mapping = {1: 0, 2: 1, 3: 1, 4: 1}

    # Create windows per shot and assign to appropriate split
    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        # Determine which split this shot belongs to
        if shot_id in train_shots:
            target_windows = train_windows
            target_labels = train_labels
        elif shot_id in val_shots:
            target_windows = val_windows
            target_labels = val_labels
        else:
            target_windows = test_windows
            target_labels = test_labels

        # Create windows for this shot
        for i in range(len(shot_indices) - window_size + 1):
            start = shot_indices[i]
            end = start + window_size

            if end > shot_indices[-1] + 1:
                break

            window = X[start:end]
            # Use LAST point label (prediction target)
            last_label = y[start + window_size - 1]

            # Check window validity
            if not np.isnan(window).any() and not np.isinf(window).any():
                target_windows.append(window)
                target_labels.append(label_mapping[int(last_label)])

    # Convert to numpy arrays
    train_windows = np.array(train_windows, dtype=np.float32)
    train_labels = np.array(train_labels)
    val_windows = np.array(val_windows, dtype=np.float32)
    val_labels = np.array(val_labels)
    test_windows = np.array(test_windows, dtype=np.float32)
    test_labels = np.array(test_labels)

    print(f"\nCreated windows:")
    print(f"  Train: {len(train_windows)} windows from {len(train_shots)} shots")
    print(f"  Val: {len(val_windows)} windows from {len(val_shots)} shots")
    print(f"  Test: {len(test_windows)} windows from {len(test_shots)} shots")

    print(f"\nBinary label distribution:")
    print(f"  Train: {Counter(train_labels)} (0=Suppressed, 1=Dithering/Mitigated/ELMing)")
    print(f"  Val: {Counter(val_labels)}")
    print(f"  Test: {Counter(test_labels)}")

    return (train_windows, train_labels,
            val_windows, val_labels,
            test_windows, test_labels)

def train_model(model, train_loader, val_loader, device, n_epochs=50):
    """Train the model"""
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    print("\nStarting training...")
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Store predictions
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(batch_y.cpu().numpy())

        # Calculate metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels_list, val_preds)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_binary_lstm_last_point_random_shot.pth')
            patience_counter = 0
            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test set"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)

            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Calculate and print test accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Calculate ROC AUC (binary)
    print("\nROC AUC Score:")
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    print(f"  Binary AUC: {auc:.4f}")

    return all_preds, all_labels, all_probs

def plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, all_probs, class_names):
    """Plot training curves, confusion matrix, and ROC curve"""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot training loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training accuracy
    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot confusion matrix (normalized)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Confusion Matrix (Binary)')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    axes[1, 1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    axes[1, 1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve (Binary Classification)')
    axes[1, 1].legend(loc='lower right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('binary_lstm_last_point_random_shot_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Results saved to 'binary_lstm_last_point_random_shot_results.png'")

import time

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("BINARY LSTM-NN Model for Plasma Classification")
    print("=" * 60)
    print("Architecture: Unidirectional LSTM → NN → Binary Classifier")
    print("Window: 150 datapoints BEFORE target point")
    print("Split: RANDOM BY SHOT NUMBER (not individual data points)")
    print("")
    print("BINARY CLASSES:")
    print("  Class 0: Suppressed")
    print("  Class 1: Dithering + Mitigated + ELMing")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    X, y, shots, features, scaler = load_and_prepare_data()

    # Create windows and split BY SHOT
    train_X, train_y, val_X, val_y, test_X, test_y = create_windows_with_random_shot_split(X, y, shots)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    # Create data loaders
    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Create model (BINARY: n_classes=2)
    model = LSTMFirstNN(n_features=len(features), n_classes=2).to(device)

    # Test forward pass speed
    print("\nTesting forward pass speed...")
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)

    start_time = time.time()
    with torch.no_grad():
        _ = model(test_batch)
    forward_time = time.time() - start_time
    print(f"Forward pass time for batch of {test_batch.shape[0]}: {forward_time:.3f} seconds")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, n_epochs=50
    )

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_binary_lstm_last_point_random_shot.pth'))

    # Evaluate on test set
    class_names = ['Suppressed', 'Dithering/Mitigated/ELMing']
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, class_names)

    # Plot results
    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, all_probs, class_names)

    # Final test accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

