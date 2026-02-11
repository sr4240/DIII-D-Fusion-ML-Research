import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import time
import os
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define prediction horizons
HORIZONS = [1, 5, 10, 20, 40, 60, 80, 100, 140, 170, 200]

class LSTMFirstNN(nn.Module):
    """
    A hybrid model with LSTM processing FIRST (for temporal patterns)
    followed by NN layers (for feature transformation).
    Uses 150 datapoints BEFORE the classification point.
    Unidirectional LSTM only (not bidirectional).
    """
    def __init__(self, n_features, n_classes=4, lstm_hidden=128, nn_hidden_sizes=[256, 128]):
        super(LSTMFirstNN, self).__init__()

        # LSTM processes the raw temporal data FIRST
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.2
        )

        lstm_output_size = lstm_hidden

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

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        batch_size, n_features, seq_len = x.shape

        # Transpose for LSTM: (batch_size, sequence_length, n_features)
        x = x.transpose(1, 2)

        # STEP 1: LSTM processes the temporal sequence
        lstm_output, _ = self.lstm(x)

        # STEP 2: Apply attention to aggregate temporal information
        attention = self.attention_weights(lstm_output)
        attended_features = torch.sum(lstm_output * attention, dim=1)

        # STEP 3: Process the final LSTM hidden state through NN
        final_hidden = lstm_output[:, -1, :]

        # Process through NN layers
        nn_features = self.nn_layers(final_hidden)

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
    print(f"Label distribution: {Counter(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, shots, selected_features, scaler

def split_shots(shots, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split unique shot IDs into train/val/test sets.
    This ensures no data leakage between sets.
    """
    unique_shots = np.unique(shots)
    np.random.seed(seed)
    np.random.shuffle(unique_shots)
    
    n_shots = len(unique_shots)
    train_size = int(train_ratio * n_shots)
    val_size = int(val_ratio * n_shots)
    
    train_shots = set(unique_shots[:train_size])
    val_shots = set(unique_shots[train_size:train_size + val_size])
    test_shots = set(unique_shots[train_size + val_size:])
    
    print(f"\nShot-based split:")
    print(f"  Train shots: {len(train_shots)}")
    print(f"  Val shots: {len(val_shots)}")
    print(f"  Test shots: {len(test_shots)}")
    
    return train_shots, val_shots, test_shots

def create_windows_single_horizon(X, y, shots, window_size=150, horizon=1):
    """
    Create windows where the label is 'horizon' timesteps into the future.
    Now also returns the shot ID for each window to enable consistent splitting.
    
    Parameters:
    - X: feature array
    - y: labels
    - shots: shot identifiers
    - window_size: number of timesteps in input window (default 150)
    - horizon: how many timesteps ahead to predict (1, 5, 10, etc.)
    
    Returns:
    - windows: array of input windows
    - labels: array of target labels
    - window_shots: array of shot IDs corresponding to each window
    """
    print(f"Creating windows for horizon t+{horizon} (window size: {window_size})...")

    windows = []
    labels_list = []
    window_shots = []

    # Create windows per shot
    for shot_id in np.unique(shots):
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        # Need enough points for window + horizon
        if len(shot_indices) < window_size + horizon:
            continue

        for i in range(len(shot_indices) - window_size - horizon + 1):
            start = shot_indices[i]
            end = start + window_size
            target_idx = end + horizon - 1  # Label at t+horizon

            # Ensure we don't go beyond shot boundaries
            if target_idx > shot_indices[-1]:
                break

            window = X[start:end]
            target_label = y[target_idx]

            # Check window validity
            if not np.isnan(window).any() and not np.isinf(window).any() and not np.isnan(target_label):
                windows.append(window)
                labels_list.append(target_label)
                window_shots.append(shot_id)

    windows = np.array(windows, dtype=np.float32)
    labels_array = np.array(labels_list)
    window_shots = np.array(window_shots)

    print(f"Created {len(windows)} valid windows for horizon t+{horizon}")

    # Map labels to 0-3 range
    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    labels_array = np.array([label_mapping[int(l)] for l in labels_array])

    print(f"Label distribution for t+{horizon}: {Counter(labels_array)}")

    return windows, labels_array, window_shots

def split_windows_by_shots(windows, labels, window_shots, train_shots, val_shots, test_shots):
    """
    Split windows into train/val/test based on shot membership.
    This ensures consistent splits across all horizons.
    """
    train_mask = np.array([shot in train_shots for shot in window_shots])
    val_mask = np.array([shot in val_shots for shot in window_shots])
    test_mask = np.array([shot in test_shots for shot in window_shots])
    
    train_X = windows[train_mask]
    train_y = labels[train_mask]
    
    val_X = windows[val_mask]
    val_y = labels[val_mask]
    
    test_X = windows[test_mask]
    test_y = labels[test_mask]
    
    return train_X, train_y, val_X, val_y, test_X, test_y

def train_model(model, train_loader, val_loader, device, horizon, n_epochs=50, patience=10):
    """Train the model for a specific horizon"""
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nTraining model for horizon t+{horizon}...")
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

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_horizon_{horizon}.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    return history, best_val_acc

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

    # Calculate test accuracy
    test_acc = accuracy_score(all_labels, all_preds)

    return all_preds, all_labels, all_probs, test_acc

def plot_horizon_comparison(results, horizons, save_path='horizon_comparison.png'):
    """Plot comparison of model performance across different horizons"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract metrics
    train_accs = [results[h]['train_acc_final'] for h in horizons]
    val_accs = [results[h]['val_acc_best'] for h in horizons]
    test_accs = [results[h]['test_acc'] for h in horizons]
    
    # Plot 1: Accuracy vs Horizon
    axes[0, 0].plot(horizons, train_accs, 'o-', label='Train', linewidth=2, markersize=8)
    axes[0, 0].plot(horizons, val_accs, 's-', label='Validation', linewidth=2, markersize=8)
    axes[0, 0].plot(horizons, test_accs, '^-', label='Test', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Model Accuracy vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Test Accuracy with error bars (showing train-test gap)
    test_accs_array = np.array(test_accs)
    train_test_gap = np.array(train_accs) - np.array(test_accs)
    axes[0, 1].errorbar(horizons, test_accs, yerr=train_test_gap, 
                       fmt='o-', linewidth=2, markersize=8, capsize=5)
    axes[0, 1].set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    axes[0, 1].set_ylabel('Test Accuracy', fontsize=12)
    axes[0, 1].set_title('Test Accuracy with Train-Test Gap', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Plot 3: Sample sizes
    sample_sizes = [results[h]['n_samples'] for h in horizons]
    axes[1, 0].bar(range(len(horizons)), sample_sizes, color='steelblue', alpha=0.7)
    axes[1, 0].set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    axes[1, 0].set_ylabel('Number of Training Samples', fontsize=12)
    axes[1, 0].set_title('Available Training Samples per Horizon', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(horizons)))
    axes[1, 0].set_xticklabels(horizons, rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Accuracy degradation
    acc_degradation = 100 * (np.array(test_accs) - test_accs[0]) / test_accs[0]
    axes[1, 1].plot(horizons, acc_degradation, 'ro-', linewidth=2, markersize=8)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy Change from t+1 (%)', fontsize=12)
    axes[1, 1].set_title('Accuracy Degradation vs Baseline (t+1)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to '{save_path}'")
    
def plot_confusion_matrices(results, horizons, class_names, save_path='confusion_matrices.png'):
    """Plot confusion matrices for selected horizons"""
    
    # Select subset of horizons to display
    display_horizons = [1, 10, 40, 100, 200]
    display_horizons = [h for h in display_horizons if h in horizons]
    
    n_horizons = len(display_horizons)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, horizon in enumerate(display_horizons):
        cm = results[horizon]['confusion_matrix']
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], cbar_kws={'label': 'Normalized Count'})
        
        test_acc = results[horizon]['test_acc']
        axes[idx].set_title(f'Horizon t+{horizon} (Acc: {test_acc:.3f})', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
    
    # Hide extra subplots
    for idx in range(n_horizons, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to '{save_path}'")

def save_results_summary(results, horizons, save_path='results_summary.txt'):
    """Save detailed results summary to text file"""
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-HORIZON PLASMA CLASSIFICATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Prediction Horizons: {horizons}\n")
        f.write(f"Window Size: 150 timesteps\n")
        f.write(f"Architecture: LSTM-First-NN (Unidirectional)\n")
        f.write(f"Split Method: Shot-based (no data leakage)\n\n")
        
        f.write("-"*80 + "\n")
        f.write(f"{'Horizon':<10} {'Samples':<12} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}\n")
        f.write("-"*80 + "\n")
        
        for h in horizons:
            f.write(f"t+{h:<8} {results[h]['n_samples']:<12} "
                   f"{results[h]['train_acc_final']:<12.4f} "
                   f"{results[h]['val_acc_best']:<12.4f} "
                   f"{results[h]['test_acc']:<12.4f}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Detailed per-class results for each horizon
        class_names = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']
        
        for h in horizons:
            f.write(f"\n{'='*80}\n")
            f.write(f"HORIZON t+{h} - DETAILED RESULTS\n")
            f.write(f"{'='*80}\n")
            f.write(f"Test Accuracy: {results[h]['test_acc']:.4f}\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-"*80 + "\n")
            
            report = results[h]['classification_report']
            for class_name in class_names:
                if class_name in report:
                    metrics = report[class_name]
                    f.write(f"{class_name:<15} "
                           f"{metrics['precision']:<12.4f} "
                           f"{metrics['recall']:<12.4f} "
                           f"{metrics['f1-score']:<12.4f}\n")
            
            f.write("\n")
    
    print(f"Results summary saved to '{save_path}'")

def main():
    """Main training pipeline for multi-horizon models"""
    
    print("=" * 80)
    print("MULTI-HORIZON PLASMA CLASSIFICATION (CONSISTENT SPLITS)")
    print("=" * 80)
    print(f"Training separate models for horizons: {HORIZONS}")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    X, y, shots, features, scaler = load_and_prepare_data()
    
    # CRITICAL: Split shots ONCE before creating any windows
    # This ensures all horizons use the same train/val/test shots
    train_shots, val_shots, test_shots = split_shots(shots, train_ratio=0.7, val_ratio=0.15, seed=42)
    
    # Store results for all horizons
    all_results = {}
    
    # Class names
    class_names = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']
    
    # Train a separate model for each horizon
    for horizon in HORIZONS:
        print(f"\n{'='*80}")
        print(f"PROCESSING HORIZON: t+{horizon}")
        print(f"{'='*80}")
        
        # Create windows for this specific horizon
        windows, labels, window_shots = create_windows_single_horizon(
            X, y, shots, 
            window_size=150, 
            horizon=horizon
        )
        
        if len(windows) < 100:
            print(f"WARNING: Only {len(windows)} samples for horizon t+{horizon}. Skipping...")
            continue
        
        # Split windows based on shot membership (CONSISTENT across all horizons)
        train_X, train_y, val_X, val_y, test_X, test_y = split_windows_by_shots(
            windows, labels, window_shots,
            train_shots, val_shots, test_shots
        )
        
        print(f"\nDataset sizes for horizon t+{horizon}:")
        print(f"  Train: {len(train_X)} samples")
        print(f"  Val: {len(val_X)} samples")
        print(f"  Test: {len(test_X)} samples")
        print(f"  Train label distribution: {Counter(train_y)}")
        print(f"  Test label distribution: {Counter(test_y)}")
        
        # Create data loaders
        train_dataset = PlasmaDataset(train_X, train_y)
        val_dataset = PlasmaDataset(val_X, val_y)
        test_dataset = PlasmaDataset(test_X, test_y)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Create model
        model = LSTMFirstNN(n_features=len(features), n_classes=4).to(device)
        
        # Train model
        history, best_val_acc = train_model(
            model, train_loader, val_loader, device, horizon, 
            n_epochs=50, patience=10
        )
        
        # Load best model
        model.load_state_dict(torch.load(f'best_model_horizon_{horizon}.pth'))
        
        # Evaluate on test set
        all_preds, all_labels, all_probs, test_acc = evaluate_model(
            model, test_loader, device, class_names
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Get classification report as dict
        report_dict = classification_report(all_labels, all_preds, 
                                           target_names=class_names, 
                                           output_dict=True, 
                                           zero_division=0)
        
        # Store results
        all_results[horizon] = {
            'n_samples': len(train_X),
            'train_acc_final': history['train_acc'][-1],
            'val_acc_best': best_val_acc,
            'test_acc': test_acc,
            'confusion_matrix': cm,
            'classification_report': report_dict,
            'history': history,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        print(f"\nResults for horizon t+{horizon}:")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Train-Test Gap: {history['train_acc'][-1] - test_acc:.4f}")
    
    # Generate comparison plots and summaries
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS AND SUMMARIES")
    print(f"{'='*80}")
    
    completed_horizons = list(all_results.keys())
    
    plot_horizon_comparison(all_results, completed_horizons, 
                           save_path='horizon_comparison.png')
    
    plot_confusion_matrices(all_results, completed_horizons, class_names,
                           save_path='confusion_matrices.png')
    
    save_results_summary(all_results, completed_horizons,
                        save_path='results_summary.txt')
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Horizon':<10} {'Test Accuracy':<15} {'Samples':<12}")
    print("-"*40)
    for h in completed_horizons:
        print(f"t+{h:<8} {all_results[h]['test_acc']:<15.4f} {all_results[h]['n_samples']:<12}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  - best_model_horizon_X.pth (for each horizon)")
    print("  - horizon_comparison.png")
    print("  - confusion_matrices.png")
    print("  - results_summary.txt")
    print("\nKey improvement: Shot-based splitting ensures no data leakage")
    print("and enables valid comparisons across all prediction horizons.")

if __name__ == "__main__":
    main()