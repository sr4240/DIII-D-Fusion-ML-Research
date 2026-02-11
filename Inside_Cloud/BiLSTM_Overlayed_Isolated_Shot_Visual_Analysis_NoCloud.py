"""
Overlayed Isolated Shot Visualization Script with BiLSTM Classification

This script visualizes BiLSTM predictions for a specified shot.
The shot is completely excluded from the training data to provide a realistic assessment.

HOW TO USE:
1. Change the SHOT_NUMBER variable below to the shot you want to analyze
2. Run the script: python BiLSTM_Overlayed_Isolated_Shot_Visual_Analysis_NoCloud.py
3. The script will generate visualization files showing BiLSTM predictions
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

plt.close('all')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS AS NEEDED
# =============================================================================

# Shot number to analyze (change this to any shot number you want to visualize)
# Examples: 169501, 169472, 169503, etc.
SHOT_NUMBER = 169472

# BiLSTM parameters
WINDOW_SIZE = 150  # Window size for BiLSTM model
BATCH_SIZE = 128
N_EPOCHS = 50

# Output file prefix (will be automatically updated based on shot number)
OUTPUT_PREFIX = f"bilstm_overlayed_shot_{SHOT_NUMBER}"

# =============================================================================

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LSTMFirstNN(nn.Module):
    """
    A hybrid model with LSTM processing FIRST (for temporal patterns)
    followed by NN layers (for feature transformation).
    """
    def __init__(self, n_features, n_classes=4, lstm_hidden=128, nn_hidden_sizes=[256, 128]):
        super(LSTMFirstNN, self).__init__()

        # LSTM processes the raw temporal data FIRST
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # After LSTM, we have temporal features
        lstm_output_size = lstm_hidden * 2  # Bidirectional

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
        lstm_output, (hidden, cell) = self.lstm(x)

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

def load_and_prepare_data_excluding_shot(exclude_shot_number=SHOT_NUMBER):
    """Load data and prepare it while excluding the specified shot from training"""
    print("Loading plasma data...")

    # Try multiple possible paths for the plasma data file
    possible_paths = [
        '../plasma_data.csv',  # Relative path from Inside_Cloud directory
        'plasma_data.csv',     # Current directory
        '/mnt/homes/sr4240/my_folder/plasma_data.csv'  # Absolute path
    ]

    df = None
    for path in possible_paths:
        try:
            print(f"Trying to load from: {path}")
            df = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            print(f"Loaded {len(df)} rows")
            break
        except FileNotFoundError:
            print(f"File not found at: {path}")
            continue

    if df is None:
        raise FileNotFoundError("Could not find plasma_data.csv in any of the expected locations")

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Add an index column to keep track of original row numbers
    df['original_index'] = df.index

    # Select features
    selected_features = ['iln3iamp', 'betan', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed']
    target_column = 'state'

    print(f"Using {len(selected_features)} features: {selected_features}")

    # Clean the dataframe
    df_cleaned = df.dropna(subset=selected_features, how='any')
    df_cleaned = df_cleaned[df_cleaned['n'] == 3]
    df_cleaned = df_cleaned[df_cleaned['state'] != 0]

    # CRITICAL: Exclude the target shot from training data
    print(f"Excluding shot {exclude_shot_number} from training data...")
    df_training = df_cleaned[df_cleaned['shot'] != exclude_shot_number].copy()
    df_target_shot = df_cleaned[df_cleaned['shot'] == exclude_shot_number].copy()

    print(f"Training data: {len(df_training)} rows (excluding shot {exclude_shot_number})")
    print(f"Target shot data: {len(df_target_shot)} rows (shot {exclude_shot_number})")

    return df_cleaned, df_training, df_target_shot, selected_features

def create_windows_with_random_split(X, y, shots, window_size=WINDOW_SIZE):
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

def train_bilstm_model(df_training, selected_features, device):
    """Train the BiLSTM model on data excluding target shot"""
    print("Training BiLSTM model on data excluding target shot...")

    # Sort by shot and time
    df_sorted = df_training.sort_values(['shot', 'time']).reset_index(drop=True)

    # Extract features and labels
    X = df_sorted[selected_features].values
    y = df_sorted['state'].values
    shots = df_sorted['shot'].values

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

    # Create windows and split
    train_X, train_y, val_X, val_y, test_X, test_y = create_windows_with_random_split(X_scaled, y, shots)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    # Create data loaders
    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = LSTMFirstNN(n_features=len(selected_features), n_classes=4).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    # Training history
    best_val_acc = 0.0
    best_model_state = model.state_dict().copy()
    patience_counter = 0
    max_patience = 10

    print("\nStarting training...")
    for epoch in range(N_EPOCHS):
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

        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model from memory
    print("\nLoading best model from training session...")
    model.load_state_dict(best_model_state)

    return model, scaler

def check_shot_exists(df, shot_number):
    """Check if the specified shot exists in the dataset"""
    shot_exists = (df['shot'] == shot_number).any()
    if not shot_exists:
        available_shots = sorted(df['shot'].unique())
        print(f"ERROR: Shot {shot_number} not found in the dataset!")
        print(f"Available shots: {available_shots[:10]}..." if len(available_shots) > 10 else f"Available shots: {available_shots}")
        return False
    return True

def extract_shot_data(df, shot_number=169501):
    """Extract all time series data for the specified shot"""
    print(f"Extracting data for shot {shot_number}...")

    # Check if shot exists first
    if not check_shot_exists(df, shot_number):
        raise ValueError(f"Shot {shot_number} not found in dataset")

    # Filter data for the specific shot
    shot_data = df[df['shot'] == shot_number].copy()

    # Sort by time to ensure proper time series order
    shot_data = shot_data.sort_values('time').reset_index(drop=True)

    print(f"Found {len(shot_data)} time points for shot {shot_number}")
    print(f"Time range: {shot_data['time'].min()} to {shot_data['time'].max()}")
    print(f"State distribution: {shot_data['state'].value_counts().to_dict()}")

    return shot_data

def create_shot_windows(shot_data, selected_features, window_size=WINDOW_SIZE):
    """Create sliding windows for the shot data for BiLSTM prediction"""
    print("Creating windows for shot prediction...")

    X = shot_data[selected_features].values

    windows = []
    window_indices = []
    center_idx = window_size // 2

    # Create windows with sliding window approach
    for i in range(len(X) - window_size + 1):
        window = X[i:i+window_size]

        # Check window validity
        if not np.isnan(window).any() and not np.isinf(window).any():
            windows.append(window)
            window_indices.append(i + center_idx)  # Index of center point

    if len(windows) == 0:
        raise ValueError("No valid windows could be created for this shot")

    windows = np.array(windows, dtype=np.float32)
    print(f"Created {len(windows)} valid windows for prediction")

    return windows, window_indices

def make_predictions_on_shot(model, shot_data, selected_features, scaler, device):
    """Make predictions on the shot data using BiLSTM model"""
    print("Making predictions on isolated shot data...")

    # Prepare features for prediction - need to scale them
    X_shot = shot_data[selected_features].values
    X_shot_scaled = scaler.transform(X_shot)

    # Create windows for the shot
    windows, window_indices = create_shot_windows(shot_data, selected_features, WINDOW_SIZE)

    # Scale the windows
    scaled_windows = []
    for window in windows:
        scaled_window = scaler.transform(window)
        scaled_windows.append(scaled_window)
    scaled_windows = np.array(scaled_windows, dtype=np.float32)

    # Create dataset and loader
    # Create dummy labels (not used for prediction)
    dummy_labels = np.zeros(len(scaled_windows))
    shot_dataset = PlasmaDataset(scaled_windows, dummy_labels)
    shot_loader = DataLoader(shot_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Make predictions
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_X, _ in shot_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Map predictions back to original labels (0-3 -> 1-4)
    label_reverse_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
    all_preds = np.array([label_reverse_mapping[p] for p in all_preds])

    # Initialize prediction columns with NaN
    shot_data['predicted_state'] = np.nan
    shot_data['prediction_confidence'] = np.nan
    shot_data['prediction_correct'] = np.nan

    # Assign predictions to corresponding center points
    for i, window_idx in enumerate(window_indices):
        shot_data.loc[window_idx, 'predicted_state'] = all_preds[i]
        shot_data.loc[window_idx, 'prediction_confidence'] = np.max(all_probs[i])

    # Calculate prediction correctness only where we have predictions
    valid_pred_mask = ~shot_data['predicted_state'].isna()
    shot_data.loc[valid_pred_mask, 'prediction_correct'] = (
        shot_data.loc[valid_pred_mask, 'state'] == shot_data.loc[valid_pred_mask, 'predicted_state']
    )

    # Calculate accuracy for this shot (only on points with predictions)
    shot_accuracy = shot_data.loc[valid_pred_mask, 'prediction_correct'].mean()
    print(f"Prediction accuracy for isolated shot {shot_data['shot'].iloc[0]}: {shot_accuracy:.3f}")
    print(f"Predicted {valid_pred_mask.sum()} out of {len(shot_data)} points")

    return shot_data

def create_overlayed_visualization(shot_data):
    """Create comprehensive visualization"""
    print("Creating overlayed visualization...")

    # Create figure with subplots (3 instead of 4)
    fig, axes = plt.subplots(3, 1, figsize=(24, 18))
    fig.suptitle(f'Plasma State Classification Analysis for Shot {SHOT_NUMBER} - BiLSTM (Split Based on Shot)',
                 fontsize=20, fontweight='bold', y=0.995)

    # Define colors for states
    state_colors = {1: '#2E8B57', 2: '#FFD700', 3: '#FF6347', 4: '#DC143C'}  # Green, Gold, Tomato, Crimson
    state_names = {1: 'Suppressed', 2: 'Dithering', 3: 'Mitigated', 4: 'ELMing'}

    time = shot_data['time']
    fs04 = shot_data['fs04']
    states = shot_data['state']

    # Plot 1: fs04 time series with actual states
    ax1 = axes[0]
    ax1.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')

    # Color background by actual state
    for state in [1, 2, 3, 4]:
        state_mask = states == state
        if state_mask.any():
            ax1.fill_between(time, fs04.min(), fs04.max(),
                           where=state_mask, alpha=0.3,
                           color=state_colors[state],
                           label=f'Actual: {state_names[state]}')

    ax1.set_ylabel('fs04 Signal (a.u.)', fontsize=16)
    ax1.set_title('(a) Observed Plasma States', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Plot 2: fs04 time series with predicted states
    ax2 = axes[1]
    predicted_states = shot_data['predicted_state']

    ax2.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')

    # Color background by predicted state (only where predictions exist)
    for state in [1, 2, 3, 4]:
        state_mask = predicted_states == state
        if state_mask.any():
            ax2.fill_between(time, fs04.min(), fs04.max(),
                           where=state_mask, alpha=0.3,
                           color=state_colors[state],
                           label=f'Predicted: {state_names[state]}')

    ax2.set_ylabel('fs04 Signal (a.u.)', fontsize=16)
    ax2.set_title('(b) Predicted Plasma States', fontsize=18, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    # Plot 3: fs04 with correct/incorrect predictions overlay (without confidence scatter)
    ax3 = axes[2]

    # Plot fs04 time series
    ax3.plot(time, fs04, 'k-', linewidth=2, alpha=0.8, label='fs04')

    # Create mask for correct/incorrect predictions (only where predictions exist)
    valid_pred = ~shot_data['prediction_correct'].isna()
    correct_mask = shot_data['prediction_correct'] == True
    incorrect_mask = shot_data['prediction_correct'] == False

    # Overlay correct predictions (green)
    if correct_mask.any():
        ax3.fill_between(time, fs04.min(), fs04.max(),
                        where=correct_mask, alpha=0.4,
                        color='green', label='Correct Predictions')

    # Overlay incorrect predictions (red)
    if incorrect_mask.any():
        ax3.fill_between(time, fs04.min(), fs04.max(),
                        where=incorrect_mask, alpha=0.4,
                        color='red', label='Incorrect Predictions')

    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04 Signal (a.u.)', fontsize=16)
    ax3.set_title('(c) Prediction Accuracy Analysis', fontsize=18, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)

    # Ensure all plots have the same x-axis limits for consistent width
    xlim_shared = (time.min(), time.max())
    ax1.set_xlim(xlim_shared)
    ax2.set_xlim(xlim_shared)
    ax3.set_xlim(xlim_shared)

    # Adjust layout to prevent overlap - ensure all plots have same width
    plt.tight_layout(rect=[0, 0, 0.88, 0.94])  # Leave space for legends on right and title at top
    plt.subplots_adjust(hspace=0.3, bottom=0.06, top=0.92)

    return fig


def print_detailed_statistics(shot_data):
    """Print detailed statistics about predictions"""
    print("\n" + "="*70)
    print(f"ANALYSIS FOR SHOT {shot_data['shot'].iloc[0]}")
    print(f"(Shot {shot_data['shot'].iloc[0]} was EXCLUDED from training data)")
    print("="*70)

    # Overall BiLSTM statistics (only on points with predictions)
    has_prediction = ~shot_data['prediction_correct'].isna()
    total_points = len(shot_data)
    predicted_points = has_prediction.sum()
    correct_predictions = (shot_data['prediction_correct'] == True).sum()
    accuracy = correct_predictions / predicted_points if predicted_points > 0 else 0

    print(f"\nBiLSTM Prediction Statistics:")
    print(f"  Total time points: {total_points}")
    print(f"  Points with predictions: {predicted_points} ({100*predicted_points/total_points:.1f}%)")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Incorrect predictions: {predicted_points - correct_predictions}")
    print(f"  Prediction accuracy: {accuracy:.3f}")

    # Statistics by actual state
    print(f"\nAccuracy by Actual State:")
    for state in [1, 2, 3, 4]:
        state_mask = shot_data['state'] == state
        if state_mask.any():
            state_data = shot_data[state_mask]
            state_has_pred = ~state_data['prediction_correct'].isna()

            if state_has_pred.any():
                state_correct = (state_data['prediction_correct'] == True).sum()
                state_total = state_has_pred.sum()
                state_accuracy = state_correct / state_total

                state_name = {1: 'Suppressed', 2: 'Dithering', 3: 'Mitigated', 4: 'ELMing'}[state]
                print(f"  {state_name} (State {state}):")
                print(f"    Accuracy: {state_correct}/{state_total} ({state_accuracy:.3f})")

def main():
    """Main function to run the complete analysis"""
    print(f"Starting BiLSTM Overlayed Isolated Shot {SHOT_NUMBER} Visualization Analysis...")
    print("="*70)
    print(f"IMPORTANT: Shot {SHOT_NUMBER} will be EXCLUDED from training data")
    print("="*70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data and split into training (excluding target shot) and target shot
    df_cleaned, df_training, df_target_shot, selected_features = load_and_prepare_data_excluding_shot(SHOT_NUMBER)

    # Train BiLSTM model (excluding target shot)
    model, feature_scaler = train_bilstm_model(df_training, selected_features, device)

    # Check if shot exists
    if not check_shot_exists(df_cleaned, SHOT_NUMBER):
        print("\nTo change the shot number, edit the SHOT_NUMBER variable at the top of this script.")
        return None

    # Extract shot data
    shot_data = extract_shot_data(df_cleaned, shot_number=SHOT_NUMBER)

    # Make BiLSTM predictions
    shot_data = make_predictions_on_shot(model, shot_data, selected_features, feature_scaler, device)

    # Create visualizations
    fig1 = create_overlayed_visualization(shot_data)

    # Print detailed statistics
    print_detailed_statistics(shot_data)

    # Create output directory if it doesn't exist
    output_dir = 'BiLSTM_Overlayed_Analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated directory: {output_dir}")

    # Save plot
    comprehensive_filename = os.path.join(output_dir, f'{OUTPUT_PREFIX}_comprehensive_overlayed.png')

    fig1.savefig(comprehensive_filename, dpi=800, bbox_inches='tight')
    print(f"\nPlot saved as:")
    print(f"  - {comprehensive_filename}")

    return shot_data

if __name__ == "__main__":
    shot_data = main()
