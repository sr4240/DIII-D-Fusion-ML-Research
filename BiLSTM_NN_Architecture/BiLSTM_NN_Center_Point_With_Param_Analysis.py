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
    This avoids the slow loop while preserving temporal structure.
    """
    def __init__(self, n_features, n_classes=4, lstm_hidden=128, nn_hidden_sizes=[256, 128]):
        super(LSTMFirstNN, self).__init__()

        # LSTM processes the raw temporal data FIRST
        # This preserves temporal relationships efficiently
        self.lstm = nn.LSTM(
            input_size=n_features,  # Direct input of raw features
            hidden_size=lstm_hidden,
            num_layers=2,  # Deeper LSTM for better temporal learning
            batch_first=True,
            bidirectional=True,  # Bidirectional for better temporal understanding
            dropout=0.2
        )

        # After LSTM, we have temporal features
        lstm_output_size = lstm_hidden * 2  # Bidirectional

        # NN layers process the LSTM output
        # No loops needed - we can process the entire LSTM output at once!
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
        print(f"LSTM-First-NN Model Parameter Count:")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nParameters by component:")
        print(f"  - LSTM layers: {lstm_params:,} ({lstm_params/total_params*100:.1f}%)")
        print(f"  - NN layers: {nn_params:,} ({nn_params/total_params*100:.1f}%)")
        print(f"  - Attention: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
        print(f"  - Classifier: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Architecture: LSTM → NN (no loops, preserves temporal structure)")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        batch_size, n_features, seq_len = x.shape

        # Transpose for LSTM: (batch_size, sequence_length, n_features)
        x = x.transpose(1, 2)

        # STEP 1: LSTM processes the temporal sequence
        lstm_output, (hidden, cell) = self.lstm(x)
        # lstm_output shape: (batch_size, seq_len, lstm_hidden*2)

        # STEP 2: Apply attention to aggregate temporal information
        attention = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        attended_features = torch.sum(lstm_output * attention, dim=1)  # (batch_size, lstm_hidden*2)

        # STEP 3: Process the final LSTM hidden state through NN
        # Take the last hidden state from both directions
        final_hidden = lstm_output[:, -1, :]  # (batch_size, lstm_hidden*2)

        # Process through NN layers (no loop needed!)
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
                         'tritop', 'fs04_max_smoothed']
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
            torch.save(model.state_dict(), 'best_lstm_first_nn.pth')
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
    plt.savefig('lstm_first_nn_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Results saved to 'lstm_first_nn_results.png'")

import time

def analyze_model_parameters(model):
    """
    Analyze model parameters to count how many have absolute values
    less than specific thresholds (0.01 and 0.001)
    """
    print("\n" + "="*60)
    print("PARAMETER SPARSITY ANALYSIS")
    print("="*60)

    # Thresholds to check
    thresholds = [0.01, 0.001]

    # Overall statistics
    total_params = 0
    threshold_counts = {thresh: 0 for thresh in thresholds}

    # Per-component statistics
    component_stats = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Get parameter values
            param_data = param.data.cpu().numpy().flatten()
            
            # Normalize parameters first (standard normalization)
            param_mean = np.mean(param_data)
            param_std = np.std(param_data)
            if param_std > 0:
                param_normalized = (param_data - param_mean) / param_std
            
            param_abs = np.abs(param_normalized)

            # Count total parameters
            n_params = len(param_data)
            total_params += n_params

            # Count parameters below each threshold
            component_stats[name] = {
                'total': n_params,
                'mean': np.mean(param_abs),
                'std': np.std(param_abs),
                'min': np.min(param_abs),
                'max': np.max(param_abs)
            }

            for thresh in thresholds:
                count = np.sum(param_abs < thresh)
                threshold_counts[thresh] += count
                component_stats[name][f'below_{thresh}'] = count
                component_stats[name][f'below_{thresh}_pct'] = (count / n_params) * 100

    # Print overall statistics
    print(f"\nTotal trainable parameters: {total_params:,}")
    print("\nOverall Sparsity:")
    for thresh in thresholds:
        count = threshold_counts[thresh]
        pct = (count / total_params) * 100
        print(f"  Parameters with |value| < {thresh}: {count:,} ({pct:.2f}%)")

    # Print per-component statistics
    print("\nPer-Component Analysis:")
    print("-" * 60)

    # Group by component type
    component_groups = {
        'LSTM': [],
        'NN Layers': [],
        'Attention': [],
        'Classifier': []
    }

    for name in component_stats.keys():
        if 'lstm' in name:
            component_groups['LSTM'].append(name)
        elif 'nn_layers' in name:
            component_groups['NN Layers'].append(name)
        elif 'attention' in name:
            component_groups['Attention'].append(name)
        elif 'classifier' in name:
            component_groups['Classifier'].append(name)

    for group_name, param_names in component_groups.items():
        if not param_names:
            continue

        print(f"\n{group_name}:")
        group_total = sum(component_stats[name]['total'] for name in param_names)

        for thresh in thresholds:
            group_below = sum(component_stats[name][f'below_{thresh}'] for name in param_names)
            group_pct = (group_below / group_total) * 100 if group_total > 0 else 0
            print(f"  |value| < {thresh}: {group_below:,}/{group_total:,} ({group_pct:.2f}%)")

    # Detailed layer-by-layer breakdown
    print("\n" + "="*60)
    print("DETAILED LAYER-BY-LAYER BREAKDOWN")
    print("="*60)

    for name, stats in component_stats.items():
        print(f"\n{name}:")
        print(f"  Shape: {total_params} total parameters")
        print(f"  Statistics:")
        print(f"    Mean |value|: {stats['mean']:.6f}")
        print(f"    Std |value|: {stats['std']:.6f}")
        print(f"    Min |value|: {stats['min']:.6f}")
        print(f"    Max |value|: {stats['max']:.6f}")
        print(f"  Sparsity:")
        for thresh in thresholds:
            count = stats[f'below_{thresh}']
            pct = stats[f'below_{thresh}_pct']
            print(f"    |value| < {thresh}: {count:,} ({pct:.2f}%)")

    print("\n" + "="*60)
    print("END OF PARAMETER ANALYSIS")
    print("="*60)

    return threshold_counts, component_stats

def main():
    """Main training pipeline"""
    print("=" * 50)
    print("LSTM-First NN Model for Plasma Classification")
    print("=" * 50)
    print("Architecture: LSTM processes temporal data → NN transforms features")
    print("Benefits: Fast processing + Preserves temporal structure")
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

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Create model
    model = LSTMFirstNN(n_features=len(features), n_classes=4).to(device)

    # Test forward pass speed
    print("\nTesting forward pass speed...")
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)

    start_time = time.time()
    with torch.no_grad():
        _ = model(test_batch)
    forward_time = time.time() - start_time
    print(f"Forward pass time for batch of {test_batch.shape[0]}: {forward_time:.3f} seconds")

    if forward_time > 0.1:
        print("  Note: Still fast! Much better than the loop-based approach (~0.9s)")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, n_epochs=50
    )

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_lstm_first_nn.pth'))

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

    # NEW: Analyze model parameters
    threshold_counts, component_stats = analyze_model_parameters(model)

if __name__ == "__main__":
    main()
