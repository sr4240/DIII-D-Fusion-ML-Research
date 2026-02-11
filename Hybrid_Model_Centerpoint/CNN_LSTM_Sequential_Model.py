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

class CNNLSTMSequential(nn.Module):
    """
    A sequential model: Input → CNN → LSTM → Dense

    Architecture:
    Raw Data → CNN → CNN Features → LSTM → LSTM Features → Dense Classifier
    """
    def __init__(self, n_features, n_classes=4, cnn_channels=32, lstm_hidden=64):
        super(CNNLSTMSequential, self).__init__()

        # CNN processes raw input first
        self.cnn = nn.Sequential(
            # First convolutional layer - input channels is n_features
            nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),

            # Second convolutional layer
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.MaxPool1d(kernel_size=2),

            # Third convolutional layer
            nn.Conv1d(cnn_channels * 2, cnn_channels * 4, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.MaxPool1d(kernel_size=2)
        )

        # Calculate CNN output dimension
        cnn_output_dim = cnn_channels * 4

        # LSTM processes CNN output
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,  # Takes CNN features as input
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Calculate LSTM output dimension
        lstm_output_dim = lstm_hidden * 2  # bidirectional

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dense classifier (takes LSTM output after pooling)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

        # Print model architecture
        total_params = sum(p.numel() for p in self.parameters())
        cnn_params = sum(p.numel() for name, p in self.named_parameters() if 'cnn' in name)
        lstm_params = sum(p.numel() for name, p in self.named_parameters() if 'lstm' in name)
        classifier_params = sum(p.numel() for name, p in self.named_parameters() if 'classifier' in name)

        print(f"\nCNN → LSTM → Dense Sequential Model initialized")
        print(f"{'='*60}")
        print(f"Architecture: Input → CNN → LSTM → Dense")
        print(f"Total parameters: {total_params:,}")
        print(f"CNN parameters: {cnn_params:,} ({cnn_params/total_params*100:.1f}%)")
        print(f"LSTM parameters: {lstm_params:,} ({lstm_params/total_params*100:.1f}%)")
        print(f"Classifier parameters: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
        print(f"{'='*60}\n")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        batch_size, n_features, seq_len = x.shape

        # ========== STEP 1: CNN Processing ==========
        # Process through CNN (already in correct format for CNN)
        cnn_features = self.cnn(x)
        # cnn_features shape: (batch_size, cnn_channels*4, reduced_seq_len)

        # ========== STEP 2: LSTM Processing ==========
        # Transpose CNN output for LSTM: (batch_size, reduced_seq_len, cnn_channels*4)
        lstm_input = cnn_features.transpose(1, 2)

        # Process through LSTM
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        # lstm_output shape: (batch_size, reduced_seq_len, lstm_hidden*2)

        # ========== STEP 3: Global Pooling ==========
        # Transpose back for pooling: (batch_size, lstm_hidden*2, reduced_seq_len)
        lstm_output_transposed = lstm_output.transpose(1, 2)

        # Apply global average pooling
        pooled_features = self.global_avg_pool(lstm_output_transposed).squeeze(-1)
        # pooled_features shape: (batch_size, lstm_hidden*2)

        # ========== STEP 4: Dense Classification ==========
        # Pass through dense classifier
        output = self.classifier(pooled_features)

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

    # Select only the 7 specified features
    important_features = ['iln3iamp', 'betan','density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed']
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
    print(f"Label distribution: {Counter(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, shots, selected_features, scaler

def create_windows_with_random_split(X, y, shots, window_size=150):
    """Create windows and perform random split"""
    print(f"Creating windows of size {window_size}...")

    windows, labels = [], []
    center_idx = window_size // 2

    # Create windows per shot
    for shot_id in np.unique(shots):
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        for i in range(len(shot_indices) - window_size + 1):
            start = shot_indices[i]
            end = start + window_size

            if end > shot_indices[-1] + 1:
                break

            window = X[start:end]
            center_label = y[start + center_idx]

            # Check window validity
            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(center_label)

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels)

    print(f"Created {len(windows)} valid windows")

    # Map labels to 0-3 range
    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    labels = np.array([label_mapping[int(l)] for l in labels])

    print(f"Mapped label distribution: {Counter(labels)}")

    # Random split
    np.random.seed(42)
    n_samples = len(windows)
    indices = np.random.permutation(n_samples)

    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return (windows[train_idx], labels[train_idx],
            windows[val_idx], labels[val_idx],
            windows[test_idx], labels[test_idx])

def train_model(model, train_loader, val_loader, device, n_epochs=50):
    """Train the model"""
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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
            torch.save(model.state_dict(), 'best_cnn_lstm_sequential.pth')
            patience_counter = 0
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

    # Calculate ROC AUC for each class
    print("\nROC AUC Scores:")
    for i, class_name in enumerate(class_names):
        if i < all_probs.shape[1]:
            class_labels = (all_labels == i).astype(int)
            if len(np.unique(class_labels)) > 1:
                auc = roc_auc_score(class_labels, all_probs[:, i])
                print(f"  {class_name}: {auc:.4f}")

    return all_preds, all_labels, all_probs

def plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names):
    """Plot training curves and confusion matrix"""

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
    axes[1, 0].set_title('Normalized Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    # Plot confusion matrix (counts)
    cm_counts = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Counts)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('cnn_lstm_sequential_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Results saved to 'cnn_lstm_sequential_results.png'")

def main():
    """Main training pipeline"""
    print("=" * 50)
    print("CNN → LSTM → Dense Sequential Model for Plasma Classification")
    print("=" * 50)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    X, y, shots, features, scaler = load_and_prepare_data()

    # Create windows and split
    train_X, train_y, val_X, val_y, test_X, test_y = create_windows_with_random_split(X, y, shots)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    # Create data loaders
    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create model
    model = CNNLSTMSequential(n_features=len(features), n_classes=4).to(device)

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, n_epochs=50
    )

    # Load best model with proper error handling
    print("\nLoading best model...")
    try:
        # Load the state dict
        checkpoint_path = 'best_cnn_lstm_sequential.pth'
        if not torch.cuda.is_available():
            # Load on CPU if CUDA is not available
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(checkpoint_path)

        # Try to load the state dict and capture any missing/unexpected keys
        load_result = model.load_state_dict(state_dict, strict=False)

        # Check for missing keys
        if load_result.missing_keys:
            print(f"WARNING: Missing keys in checkpoint: {load_result.missing_keys}")
            print("These parameters will use their initialized values.")

        # Check for unexpected keys
        if load_result.unexpected_keys:
            print(f"WARNING: Unexpected keys in checkpoint: {load_result.unexpected_keys}")
            print("These parameters from the checkpoint will be ignored.")

        # If strict loading is required, raise an error for any mismatches
        if load_result.missing_keys or load_result.unexpected_keys:
            print("\nModel architecture mismatch detected!")
            print("Current model parameters:")
            for name, param in model.named_parameters():
                print(f"  {name}: {param.shape}")

            # Ask user if they want to continue
            response = input("\nDo you want to continue with partial loading? (y/n): ")
            if response.lower() != 'y':
                raise ValueError("Model loading cancelled due to architecture mismatch.")
        else:
            print("Model loaded successfully with all keys matching!")

    except FileNotFoundError:
        print(f"ERROR: Checkpoint file '{checkpoint_path}' not found!")
        print("Training from scratch or provide a valid checkpoint path.")
        raise
    except RuntimeError as e:
        print(f"ERROR: Failed to load model due to incompatible shapes: {e}")
        print("\nExpected model architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error during model loading: {e}")
        raise

    # Evaluate on test set
    class_names = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, class_names)

    # Plot results
    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names)

    # Final test accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()