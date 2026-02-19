import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Prediction horizon in milliseconds
PREDICTION_HORIZON_MS = 50
WINDOW_SIZE = 150

# Import model architecture from the training script
class LSTMFirstNN(nn.Module):
    """LSTM-NN model for binary classification"""
    def __init__(self, n_features, n_classes=2, lstm_hidden=64, nn_hidden_sizes=[128, 64]):
        super(LSTMFirstNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.4
        )

        lstm_output_size = lstm_hidden

        nn_layers = []
        input_dim = lstm_output_size

        for hidden_size in nn_hidden_sizes:
            nn_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.45)
            ])
            input_dim = hidden_size

        self.nn_layers = nn.Sequential(*nn_layers)

        self.attention_weights = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim + lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        batch_size, n_features, seq_len = x.shape
        x = x.transpose(1, 2)

        lstm_output, (hidden, cell) = self.lstm(x)

        attention = self.attention_weights(lstm_output)
        attended_features = torch.sum(lstm_output * attention, dim=1)

        final_hidden = lstm_output[:, -1, :]
        nn_features = self.nn_layers(final_hidden)

        combined = torch.cat([nn_features, attended_features], dim=1)
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
        return self.windows[idx].T, self.labels[idx]

def load_and_prepare_data():
    """Load and preprocess the plasma data"""
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    # Select features
    important_features = ['iln3iamp', 'betan', 'density', 'li',
                         'tritop', 'fs04_past_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    # Sort by shot and time
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Calculate fs04 rate of change
    if 'fs04' in df_sorted.columns:
        fs04_values = df_sorted['fs04'].values
        times_temp = df_sorted['time'].values
        shots_temp = df_sorted['shot'].values
        
        fs04_rate_of_change = np.zeros(len(df_sorted))
        
        for shot_id in df_sorted['shot'].unique():
            shot_mask = shots_temp == shot_id
            shot_indices = np.where(shot_mask)[0]
            
            if len(shot_indices) > 1:
                fs04_diff = np.diff(fs04_values[shot_indices])
                time_diff = np.diff(times_temp[shot_indices])
                time_diff_safe = np.where(time_diff == 0, 1, time_diff)
                rate = fs04_diff / time_diff_safe
                
                fs04_rate_of_change[shot_indices[0]] = 0.0
                fs04_rate_of_change[shot_indices[1:]] = rate
        
        df_sorted['fs04_rate_of_change'] = fs04_rate_of_change
        # selected_features.append('fs04_rate_of_change')  # Removed from LSTM input features
    
    print(f"Using {len(selected_features)} features: {selected_features}")

    # Extract features, labels, times, and shots
    X = df_sorted[selected_features].values
    y = df_sorted['state'].values
    times = df_sorted['time'].values
    shots = df_sorted['shot'].values

    # Remove NaN values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(times)
    X = X[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]
    df_sorted = df_sorted[valid_mask].reset_index(drop=True)

    print(f"Data shape after cleaning: {X.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, times, shots, selected_features, scaler, df_sorted

def create_windows_for_visualization(X, y, times, shots, window_size=150, prediction_horizon_ms=50):
    """Create windows for visualization - returns windows with metadata including current states"""
    binary_label_mapping = {1: 0, 2: 0, 3: 1, 4: 1}
    
    windows = []
    labels = []
    current_states = []  # Current state at end of window
    window_times = []  # Time at end of window (prediction point)
    window_shots = []
    window_indices = []  # Original indices in dataframe
    
    unique_shots = np.unique(shots)
    
    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        shot_times = times[shot_indices]
        shot_labels = y[shot_indices]
        shot_X = X[shot_indices]

        for i in range(len(shot_indices) - window_size + 1):
            window = shot_X[i:i + window_size]
            
            window_end_time = shot_times[i + window_size - 1]
            target_time = window_end_time + prediction_horizon_ms
            
            # Get current state at end of window
            current_label = shot_labels[i + window_size - 1]
            
            future_local_idx = np.searchsorted(shot_times, target_time)
            
            if future_local_idx >= len(shot_times):
                continue
            
            future_label = shot_labels[future_local_idx]

            # Only create example if both current and future labels are valid
            if int(current_label) not in binary_label_mapping or int(future_label) not in binary_label_mapping:
                continue

            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(binary_label_mapping[int(future_label)])
                current_states.append(binary_label_mapping[int(current_label)])
                window_times.append(window_end_time)
                window_shots.append(shot_id)
                window_indices.append(shot_indices[i + window_size - 1])

    return (np.array(windows, dtype=np.float32), 
            np.array(labels),
            np.array(current_states),
            np.array(window_times),
            np.array(window_shots),
            np.array(window_indices))

def evaluate_per_shot(model, test_loader, device, window_shots, current_states, window_times, threshold=0.5):
    """Evaluate model and return predictions grouped by shot, including transition counts and timing"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_shots = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            pos_class_probs = probs[:, 1].cpu().numpy()
            preds = (pos_class_probs >= threshold).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy per shot and count transitions
    shot_accuracies = {}
    for shot_id in np.unique(window_shots):
        shot_mask = window_shots == shot_id
        shot_preds = all_preds[shot_mask]
        shot_labels = all_labels[shot_mask]
        shot_current_states = current_states[shot_mask]
        shot_times = window_times[shot_mask]
        
        if len(shot_preds) > 0:
            acc = accuracy_score(shot_labels, shot_preds)
            # Count transitions (where current_state != future_label)
            transition_mask = shot_current_states != shot_labels
            n_transitions = np.sum(transition_mask)
            
            # Get times when transitions occur
            transition_times = shot_times[transition_mask] if n_transitions > 0 else np.array([])
            
            shot_accuracies[shot_id] = {
                'accuracy': acc,
                'n_samples': len(shot_preds),
                'n_transitions': n_transitions,
                'transition_times': transition_times,
                'all_times': shot_times,
                'time_range': (shot_times.min(), shot_times.max()) if len(shot_times) > 0 else (0, 0),
                'predictions': shot_preds,
                'labels': shot_labels,
                'current_states': shot_current_states,
                'indices': np.where(shot_mask)[0]
            }
    
    return shot_accuracies, all_preds, all_labels

def plot_shot_visualization(df, shot_id, shot_data, title_suffix="", save_path=None):
    """Create publication-quality visualization for a shot"""
    
    # Get shot data from dataframe
    shot_df = df[df['shot'] == shot_id].copy()
    shot_df = shot_df.sort_values('time').reset_index(drop=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Shot {shot_id} - {title_suffix}', fontsize=16, fontweight='bold', y=0.995)
    
    time = shot_df['time'].values
    fs04 = shot_df['fs04'].values if 'fs04' in shot_df.columns else shot_df['fs04_past_max_smoothed'].values
    states = shot_df['state'].values
    
    # Binary mapping for states
    binary_states = np.array([0 if s in [1, 2] else 1 if s in [3, 4] else -1 for s in states])
    
    # Predictions are made at the end of each window, predicting state 50ms in the future
    # So we need to map predictions to the FUTURE time point (pred_time + 50ms), not the window end time
    PREDICTION_HORIZON_MS = 50
    
    # Map predictions to time points (at the future time they're predicting)
    pred_binary = np.full(len(shot_df), -1, dtype=float)
    true_binary = np.full(len(shot_df), -1, dtype=float)
    
    for idx, (pred, true, window_end_time) in enumerate(zip(shot_data['predictions'], 
                                                             shot_data['labels'],
                                                             shot_data['times'])):
        # Prediction is for the state 50ms in the future from window end time
        future_time = window_end_time + PREDICTION_HORIZON_MS
        
        # Find closest time point in shot_df to the future time
        time_idx = np.argmin(np.abs(shot_df['time'].values - future_time))
        if time_idx < len(pred_binary):
            pred_binary[time_idx] = pred
            true_binary[time_idx] = true
    
    # Plot 1: fs04 with actual binary states
    ax1 = axes[0]
    ax1.plot(time, fs04, 'k-', linewidth=1.5, alpha=0.8, label='fs04')
    
    # Color background by actual binary state
    for state_val, color, label in [(0, '#4A90E2', 'Suppressed'), 
                                     (1, '#E24A4A', 'Mitigated/Dithering/ELMing')]:
        mask = binary_states == state_val
        if mask.any():
            ax1.fill_between(time, fs04.min() * 0.98, fs04.max() * 1.02,
                           where=mask, alpha=0.25, color=color, label=f'Actual: {label}')
    
    ax1.set_ylabel('fs04 Signal', fontsize=12, fontweight='bold')
    ax1.set_title('Actual States', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(time.min(), time.max())
    
    # Plot 2: fs04 with predicted binary states
    ax2 = axes[1]
    ax2.plot(time, fs04, 'k-', linewidth=1.5, alpha=0.8, label='fs04')
    
    # Color background by predicted binary state
    for state_val, color, label in [(0, '#4A90E2', 'Suppressed'), 
                                     (1, '#E24A4A', 'Mitigated/Dithering/ELMing')]:
        mask = pred_binary == state_val
        if mask.any():
            ax2.fill_between(time, fs04.min() * 0.98, fs04.max() * 1.02,
                           where=mask, alpha=0.25, color=color, label=f'Predicted: {label}')
    
    ax2.set_ylabel('fs04 Signal', fontsize=12, fontweight='bold')
    ax2.set_title('Predicted States', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(time.min(), time.max())
    
    # Plot 3: Comparison - highlight misclassifications
    ax3 = axes[2]
    ax3.plot(time, fs04, 'k-', linewidth=1.5, alpha=0.6, label='fs04')
    
    # Highlight correct predictions
    correct_mask = (pred_binary == true_binary) & (pred_binary >= 0)
    if correct_mask.any():
        ax3.fill_between(time, fs04.min() * 0.98, fs04.max() * 1.02,
                        where=correct_mask, alpha=0.2, color='green', 
                        label='Correct Prediction')
    
    # Highlight misclassifications
    error_mask = (pred_binary != true_binary) & (pred_binary >= 0) & (true_binary >= 0)
    if error_mask.any():
        ax3.fill_between(time, fs04.min() * 0.98, fs04.max() * 1.02,
                        where=error_mask, alpha=0.4, color='red', 
                        label='Misclassification')
    
    ax3.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('fs04 Signal', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Accuracy', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(time.min(), time.max())
    
    # Add accuracy text
    if len(shot_data['predictions']) > 0:
        acc = accuracy_score(shot_data['labels'], shot_data['predictions'])
        ax3.text(0.02, 0.98, f'Accuracy: {acc:.3f}', 
                transform=ax3.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig

def main():
    """Main function to visualize best and worst shots"""
    print("=" * 60)
    print("Visualizing Best and Worst Shots")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X, y, times, shots, features, scaler, df = load_and_prepare_data()
    
    # Determine train/val/test split (same as training script)
    np.random.seed(42)
    unique_shots = np.unique(shots)
    shuffled_shots = np.random.permutation(unique_shots)
    n_shots = len(unique_shots)
    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)
    test_shots_set = set(shuffled_shots[train_size + val_size:])
    val_shots_set = set(shuffled_shots[train_size:train_size + val_size])
    
    # Create windows for all data
    print("\nCreating windows for evaluation...")
    all_windows, all_labels, all_current_states, all_times, all_shots, all_indices = create_windows_for_visualization(
        X, y, times, shots, window_size=WINDOW_SIZE, prediction_horizon_ms=PREDICTION_HORIZON_MS
    )
    
    # Filter to test shots
    test_mask = np.array([s in test_shots_set for s in all_shots])
    test_windows = all_windows[test_mask]
    test_labels = all_labels[test_mask]
    test_current_states = all_current_states[test_mask]
    test_times = all_times[test_mask]
    test_shots = all_shots[test_mask]
    test_indices = all_indices[test_mask]
    
    print(f"Test windows: {len(test_windows)}")
    
    # Create data loaders
    test_dataset = PlasmaDataset(test_windows, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    # Load model
    print("\nLoading trained model...")
    model = LSTMFirstNN(n_features=len(features), n_classes=2).to(device)
    model.load_state_dict(torch.load('best_lstm_50ms_binary_transitions.pth'))
    
    # Find optimal threshold - recalculate on validation set
    print("\nFinding optimal threshold on validation set...")
    val_mask = np.array([s in val_shots_set for s in all_shots])
    val_windows = all_windows[val_mask]
    val_labels = all_labels[val_mask]
    # Note: val_current_states not needed for threshold finding
    
    val_dataset = PlasmaDataset(val_windows, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    
    # Find optimal threshold
    print("\nFinding optimal threshold on validation set...")
    val_probs = []
    val_labels_list = []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            val_probs.extend(probs[:, 1].cpu().numpy())
            val_labels_list.extend(batch_y.numpy())
    
    val_probs = np.array(val_probs)
    val_labels_list = np.array(val_labels_list)
    
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (val_probs >= threshold).astype(int)
        if len(np.unique(preds)) > 1:
            f1 = f1_score(val_labels_list, preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    optimal_threshold = best_threshold
    print(f"Optimal threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
    
    # Evaluate per shot
    print("\nEvaluating predictions per shot...")
    shot_accuracies, all_preds, all_labels = evaluate_per_shot(
        model, test_loader, device, test_shots, test_current_states, test_times, threshold=optimal_threshold
    )
    
    # Visualize only shot 169503
    target_shot = 169503
    
    if target_shot not in shot_accuracies:
        print(f"\nShot {target_shot} not found in test set!")
        print(f"Available shots in test set: {sorted(shot_accuracies.keys())}")
        return
    
    shot_data = shot_accuracies[target_shot]
    
    print(f"\nVisualizing shot {target_shot}:")
    print(f"  Accuracy: {shot_data['accuracy']:.4f}")
    print(f"  Samples: {shot_data['n_samples']}")
    print(f"  Transitions: {shot_data['n_transitions']}")
    print(f"  Time range: {shot_data['time_range'][0]:.1f}-{shot_data['time_range'][1]:.1f} ms")
    
    # Prepare shot data for visualization
    def prepare_shot_data(shot_id, shot_acc_data):
        indices = shot_acc_data['indices']
        return {
            'predictions': shot_acc_data['predictions'],
            'labels': shot_acc_data['labels'],
            'times': test_times[indices]
        }
    
    # Create visualization
    print("\nCreating visualization...")
    
    target_shot_data = prepare_shot_data(target_shot, shot_data)
    target_shot_path = os.path.join(script_dir, f'lstm_shot_{target_shot}.png')
    plot_shot_visualization(
        df, target_shot, target_shot_data,
        title_suffix="LSTM Predictions",
        save_path=target_shot_path
    )
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

