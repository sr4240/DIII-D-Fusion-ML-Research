"""
Overlayed Isolated Shot Visualization Script with BiLSTM Classification and kNN Cloud Classification

This script visualizes BiLSTM predictions for a specified shot while also overlaying
the kNN cloud classification to show which points are inside or outside the training data cloud.
The shot is completely excluded from the training data to provide a realistic assessment.

The script uses a pre-trained BiLSTM model loaded from 'best_lstm_first_nn.pth' instead of
retraining each time, making the analysis much faster.

HOW TO USE:
1. Ensure 'best_lstm_first_nn.pth' model file exists in the current directory
2. Change the SHOT_NUMBER variable below to the shot you want to analyze
3. Run the script: python BiLSTM_Overlayed_Isolated_Shot_Visual_Analysis.py
4. The script will generate visualization files showing both BiLSTM predictions and kNN cloud classification

The script combines BiLSTM prediction accuracy with kNN cloud distance metrics to provide
a comprehensive view of model performance and data distribution.
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
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
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

# kNN parameters
N_NEIGHBORS = 20  # Number of nearest neighbors for kNN cloud
PERCENTILE_THRESHOLD = 95.0  # Percentile for determining cloud boundary

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

# kNN feature columns (from knn_cloud.py)
KNN_FEATURE_COLUMNS = [
    "iln3iamp",
    "betan",
    "density",
    "n_eped",
    "li",
    "tritop",
    "fs04_max_smoothed",
]

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

def load_bilstm_model(df_training, selected_features, device, model_path='best_lstm_first_nn.pth'):
    """Load the pre-trained BiLSTM model from saved parameters"""
    print(f"Loading BiLSTM model from {model_path}...")

    # Try multiple possible paths for the model file
    possible_paths = [
        model_path,  # Current directory or absolute path
        os.path.join('..', model_path),  # Parent directory
        os.path.join('/mnt/homes/sr4240/my_folder', model_path)  # Absolute path
    ]

    model_file = None
    for path in possible_paths:
        if os.path.exists(path):
            model_file = path
            print(f"Found model at: {path}")
            break

    if model_file is None:
        raise FileNotFoundError(f"Model file '{model_path}' not found in any of the expected locations: {possible_paths}")

    # Sort by shot and time
    df_sorted = df_training.sort_values(['shot', 'time']).reset_index(drop=True)

    # Extract features and labels
    X = df_sorted[selected_features].values
    y = df_sorted['state'].values

    # Remove NaN values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Label distribution: {Counter(y)}")

    # Standardize features - we need this to match the training preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create model
    model = LSTMFirstNN(n_features=len(selected_features), n_classes=4).to(device)

    # Load saved model parameters
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()  # Set to evaluation mode

    print(f"Successfully loaded model from {model_file}")

    return model, scaler

def train_knn_cloud(df_training):
    """Train kNN cloud model for determining if points are inside or outside the training data cloud"""
    print("\nTraining kNN cloud model...")

    # Select only the kNN features and drop NaN values
    knn_features_df = df_training[KNN_FEATURE_COLUMNS].dropna()

    if knn_features_df.empty:
        raise ValueError("No data available for kNN cloud after cleaning")

    # Fit imputer and standardizer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(knn_features_df.values)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_imputed)

    # Compute per-point mean distance to N_NEIGHBORS among training points
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric='euclidean')
    nn.fit(X_train_std)
    distances, indices = nn.kneighbors(X_train_std, return_distance=True)
    # distances[:, 0] corresponds to distance to self (0); drop it
    train_mean_dists = distances[:, 1:].mean(axis=1)

    # Calculate threshold at specified percentile
    threshold = float(np.percentile(train_mean_dists, PERCENTILE_THRESHOLD))

    print(f"kNN cloud statistics:")
    print(f"  - Neighbors: {N_NEIGHBORS}")
    print(f"  - Percentile: {PERCENTILE_THRESHOLD:.1f}")
    print(f"  - Threshold (mean distance at {PERCENTILE_THRESHOLD:.1f}th pct): {threshold:.6f}")

    return imputer, scaler, X_train_std, threshold

def classify_shot_with_knn_cloud(shot_data, imputer, scaler, X_train_std, threshold):
    """Classify each point in the shot as inside or outside the kNN cloud"""
    print("Classifying shot points with kNN cloud...")

    # Prepare features for kNN (handle missing values)
    shot_knn_features = shot_data[KNN_FEATURE_COLUMNS].copy()

    # Track which rows have valid kNN features
    valid_knn_mask = ~shot_knn_features.isnull().any(axis=1)

    # Initialize cloud classification column
    shot_data['in_cloud'] = np.nan
    shot_data['knn_distance'] = np.nan

    if valid_knn_mask.sum() > 0:
        # Transform valid rows
        X_shot = shot_knn_features[valid_knn_mask].values
        X_shot_imputed = imputer.transform(X_shot)
        X_shot_std = scaler.transform(X_shot_imputed)

        # Compute mean distances to training neighbors
        nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='euclidean')
        nn.fit(X_train_std)
        distances, _ = nn.kneighbors(X_shot_std, return_distance=True)
        mean_dists = distances.mean(axis=1)

        # Classify as in/out of cloud
        in_cloud = mean_dists <= threshold

        # Store results only for valid rows
        shot_data.loc[valid_knn_mask, 'in_cloud'] = in_cloud
        shot_data.loc[valid_knn_mask, 'knn_distance'] = mean_dists

        # Statistics
        below = int(in_cloud.sum())
        above = int((~in_cloud).sum())
        total = int(len(in_cloud))
        pct_below = 100.0 * below / float(total) if total > 0 else 0.0
        pct_above = 100.0 * above / float(total) if total > 0 else 0.0

        print(f"kNN cloud classification for shot {shot_data['shot'].iloc[0]}:")
        print(f"  - Inside cloud: {pct_below:.2f}% ({below}/{total})")
        print(f"  - Outside cloud: {pct_above:.2f}% ({above}/{total})")
        print(f"  - Invalid kNN features: {(~valid_knn_mask).sum()} points")
    else:
        print("Warning: No valid kNN features found in shot data")

    return shot_data

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
    """Create comprehensive visualization with kNN cloud overlay"""
    print("Creating overlayed visualization...")

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(24, 24))
    fig.suptitle(f'Isolated Shot {SHOT_NUMBER} Analysis with BiLSTM and kNN Cloud Classification',
                 fontsize=20, fontweight='bold', y=0.98)

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

    ax1.set_ylabel('fs04', fontsize=16)
    ax1.set_title('fs04 Time Series with Actual States', fontsize=18, fontweight='bold')
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

    ax2.set_ylabel('fs04', fontsize=16)
    ax2.set_title('fs04 Time Series with BiLSTM Predicted States', fontsize=18, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    # Plot 3: fs04 with correct/incorrect predictions overlay
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

    # Add markers for prediction confidence where predictions exist
    if valid_pred.any():
        confidence = shot_data.loc[valid_pred, 'prediction_confidence']
        scatter = ax3.scatter(time[valid_pred], fs04[valid_pred], c=confidence, cmap='viridis',
                             s=40, alpha=0.9, edgecolors='none', linewidth=0)

    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04', fontsize=16)
    ax3.set_title('BiLSTM Prediction Accuracy with Confidence Overlay', fontsize=18, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.12, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)

    # Plot 4: kNN distance from training cloud with correct/incorrect overlay
    ax4 = axes[3]

    # Plot kNN distance where available
    valid_knn = ~shot_data['knn_distance'].isna()
    if valid_knn.any():
        knn_distances = shot_data.loc[valid_knn, 'knn_distance']
        knn_times = time[valid_knn]

        # Plot distance line
        ax4.plot(knn_times, knn_distances, 'b-', linewidth=2, alpha=0.8, label='kNN Distance')

        # Get masks for correct/incorrect predictions
        in_cloud_mask = shot_data['in_cloud'] == True
        out_cloud_mask = shot_data['in_cloud'] == False

        # Overlay correct/incorrect predictions on the kNN distance
        # Correct predictions (green scatter)
        correct_points = shot_data.loc[valid_knn & correct_mask]
        if not correct_points.empty:
            ax4.scatter(correct_points['time'], correct_points['knn_distance'],
                       color='green', s=50, alpha=0.7, label='Correct Predictions',
                       edgecolors='darkgreen', linewidth=1)

        # Incorrect predictions (red scatter)
        incorrect_points = shot_data.loc[valid_knn & incorrect_mask]
        if not incorrect_points.empty:
            ax4.scatter(incorrect_points['time'], incorrect_points['knn_distance'],
                       color='red', s=50, alpha=0.7, label='Incorrect Predictions',
                       edgecolors='darkred', linewidth=1)

        # Add a horizontal line to indicate approximate cloud boundary
        if in_cloud_mask.any() and out_cloud_mask.any():
            in_cloud_distances = shot_data.loc[valid_knn & in_cloud_mask, 'knn_distance']
            out_cloud_distances = shot_data.loc[valid_knn & out_cloud_mask, 'knn_distance']
            if not in_cloud_distances.empty and not out_cloud_distances.empty:
                estimated_threshold = (in_cloud_distances.max() + out_cloud_distances.min()) / 2
                ax4.axhline(y=estimated_threshold, color='gray', linestyle='--',
                           linewidth=1.5, alpha=0.7, label='Cloud Boundary (est.)')

    ax4.set_xlabel('Time (ms)', fontsize=16)
    ax4.set_ylabel('kNN Distance to Training Cloud', fontsize=16)
    ax4.set_title('Distance from Training Data Cloud with BiLSTM Prediction Accuracy', fontsize=18, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, hspace=0.3, bottom=0.06, top=0.96)

    return fig

def create_detailed_analysis_plot(shot_data):
    """Create a detailed analysis plot focusing on the relationship between predictions and cloud classification"""
    print("Creating detailed analysis plot...")

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle(f'Shot {SHOT_NUMBER}: BiLSTM Predictions vs kNN Cloud Analysis\n(Shot excluded from training)',
                 fontsize=20, fontweight='bold')

    time = shot_data['time']
    fs04 = shot_data['fs04']
    actual_states = shot_data['state']
    predicted_states = shot_data['predicted_state']
    correct_predictions = shot_data['prediction_correct'] == True
    in_cloud = shot_data['in_cloud'] == True
    out_cloud = shot_data['in_cloud'] == False
    has_prediction = ~shot_data['prediction_correct'].isna()

    # Plot 1: State transitions with cloud indicators
    ax1 = axes[0]

    # Plot actual states
    ax1.step(time, actual_states, 'b-', linewidth=2, label='Actual State', where='post')

    # Plot predicted states where available
    if has_prediction.any():
        pred_time = time[has_prediction]
        pred_states = predicted_states[has_prediction]
        ax1.step(pred_time, pred_states, 'r--', linewidth=2, label='Predicted State (BiLSTM)', where='post')

    # Mark points outside cloud with vertical lines
    out_cloud_times = time[out_cloud]
    for t in out_cloud_times[::5]:  # Plot every 5th point to avoid clutter
        ax1.axvline(x=t, color='orange', alpha=0.2, linewidth=0.5)

    # Highlight incorrect predictions
    incorrect_times = time[has_prediction & ~correct_predictions]
    incorrect_actual = actual_states[has_prediction & ~correct_predictions]
    if len(incorrect_times) > 0:
        ax1.scatter(incorrect_times, incorrect_actual, color='red', s=60,
                   marker='x', label='Incorrect Predictions', zorder=5)

    ax1.set_ylabel('State', fontsize=16)
    ax1.set_title('State Transitions with Cloud Membership', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=14, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 4.5)
    ax1.set_yticks([1, 2, 3, 4])
    ax1.set_yticklabels(['Suppressed', 'Dithering', 'Mitigated', 'ELMing'])

    # Plot 2: Accuracy breakdown by cloud membership
    ax2 = axes[1]

    # Calculate accuracy metrics (only where we have predictions)
    categories = ['In Cloud\nCorrect', 'In Cloud\nIncorrect', 'Out Cloud\nCorrect', 'Out Cloud\nIncorrect']
    counts = [
        (correct_predictions & in_cloud).sum(),
        (has_prediction & ~correct_predictions & in_cloud).sum(),
        (correct_predictions & out_cloud).sum(),
        (has_prediction & ~correct_predictions & out_cloud).sum()
    ]

    colors_bar = ['green', 'red', 'lightgreen', 'lightcoral']
    bars = ax2.bar(categories, counts, color=colors_bar, edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    total_in_cloud = (has_prediction & in_cloud).sum()
    total_out_cloud = (has_prediction & out_cloud).sum()

    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        if i < 2 and total_in_cloud > 0:  # In cloud percentages
            pct = 100 * count / total_in_cloud
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        elif i >= 2 and total_out_cloud > 0:  # Out of cloud percentages
            pct = 100 * count / total_out_cloud
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Number of Points', fontsize=16)
    ax2.set_title('BiLSTM Prediction Accuracy by Cloud Membership', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Time series colored by combined classification
    ax3 = axes[2]

    # Plot fs04
    ax3.plot(time, fs04, 'k-', linewidth=1, alpha=0.3, label='fs04')

    # Create combined classification colors
    combined_colors = []
    combined_labels = []
    for i in range(len(shot_data)):
        if pd.isna(shot_data.iloc[i]['in_cloud']):
            combined_colors.append('gray')
            combined_labels.append('No kNN data')
        elif pd.isna(shot_data.iloc[i]['prediction_correct']):
            combined_colors.append('lightgray')
            combined_labels.append('No prediction')
        elif correct_predictions.iloc[i] and in_cloud.iloc[i]:
            combined_colors.append('green')
            combined_labels.append('Correct + In Cloud')
        elif correct_predictions.iloc[i] and out_cloud.iloc[i]:
            combined_colors.append('lightgreen')
            combined_labels.append('Correct + Out Cloud')
        elif has_prediction.iloc[i] and not correct_predictions.iloc[i] and in_cloud.iloc[i]:
            combined_colors.append('red')
            combined_labels.append('Incorrect + In Cloud')
        elif has_prediction.iloc[i] and not correct_predictions.iloc[i] and out_cloud.iloc[i]:
            combined_colors.append('orange')
            combined_labels.append('Incorrect + Out Cloud')
        else:
            combined_colors.append('lightgray')
            combined_labels.append('No prediction')

    # Scatter plot with combined classification
    scatter = ax3.scatter(time, fs04, c=combined_colors, s=30, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add legend with unique labels
    unique_combinations = []
    unique_colors = []
    for color, label in zip(['green', 'lightgreen', 'red', 'orange', 'lightgray', 'gray'],
                           ['Correct + In Cloud', 'Correct + Out Cloud',
                            'Incorrect + In Cloud', 'Incorrect + Out Cloud', 'No prediction', 'No kNN data']):
        if color in combined_colors:
            unique_combinations.append(plt.scatter([], [], c=color, s=50, edgecolors='black', linewidth=0.5))
            unique_colors.append(label)

    ax3.legend(unique_combinations, unique_colors, fontsize=12, loc='upper right')

    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04', fontsize=16)
    ax3.set_title('Combined BiLSTM Accuracy and kNN Cloud Classification', fontsize=18, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Ensure proper layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.94, hspace=0.35)

    return fig

def print_detailed_statistics(shot_data):
    """Print detailed statistics about predictions and cloud classification"""
    print("\n" + "="*70)
    print(f"OVERLAYED ANALYSIS FOR SHOT {shot_data['shot'].iloc[0]}")
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

    # kNN Cloud statistics
    in_cloud = shot_data['in_cloud'] == True
    out_cloud = shot_data['in_cloud'] == False
    no_knn = shot_data['in_cloud'].isna()

    print(f"\nkNN Cloud Classification:")
    print(f"  Points inside cloud: {in_cloud.sum()}")
    print(f"  Points outside cloud: {out_cloud.sum()}")
    print(f"  Points without kNN features: {no_knn.sum()}")

    # Combined statistics
    print(f"\nCombined Analysis:")

    if (has_prediction & in_cloud).sum() > 0:
        in_cloud_correct = ((shot_data['prediction_correct'] == True) & in_cloud).sum()
        in_cloud_total = (has_prediction & in_cloud).sum()
        in_cloud_accuracy = in_cloud_correct / in_cloud_total
        print(f"  In-cloud accuracy: {in_cloud_correct}/{in_cloud_total} ({in_cloud_accuracy:.3f})")

    if (has_prediction & out_cloud).sum() > 0:
        out_cloud_correct = ((shot_data['prediction_correct'] == True) & out_cloud).sum()
        out_cloud_total = (has_prediction & out_cloud).sum()
        out_cloud_accuracy = out_cloud_correct / out_cloud_total
        print(f"  Out-of-cloud accuracy: {out_cloud_correct}/{out_cloud_total} ({out_cloud_accuracy:.3f})")

    # Breakdown by category
    print(f"\nDetailed Breakdown:")
    print(f"  Correct + In Cloud: {((shot_data['prediction_correct'] == True) & in_cloud).sum()}")
    print(f"  Correct + Out Cloud: {((shot_data['prediction_correct'] == True) & out_cloud).sum()}")
    print(f"  Incorrect + In Cloud: {(has_prediction & (shot_data['prediction_correct'] == False) & in_cloud).sum()}")
    print(f"  Incorrect + Out Cloud: {(has_prediction & (shot_data['prediction_correct'] == False) & out_cloud).sum()}")

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

                state_in_cloud = (state_data['in_cloud'] == True).sum()
                state_out_cloud = (state_data['in_cloud'] == False).sum()

                state_name = {1: 'Suppressed', 2: 'Dithering', 3: 'Mitigated', 4: 'ELMing'}[state]
                print(f"  {state_name} (State {state}):")
                print(f"    Accuracy: {state_correct}/{state_total} ({state_accuracy:.3f})")
                print(f"    In cloud: {state_in_cloud}, Out of cloud: {state_out_cloud}")

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

    # Load pre-trained BiLSTM model
    model, feature_scaler = load_bilstm_model(df_training, selected_features, device)

    # Train kNN cloud model
    imputer, knn_scaler, X_train_std, threshold = train_knn_cloud(df_training)

    # Check if shot exists
    if not check_shot_exists(df_cleaned, SHOT_NUMBER):
        print("\nTo change the shot number, edit the SHOT_NUMBER variable at the top of this script.")
        return None

    # Extract shot data
    shot_data = extract_shot_data(df_cleaned, shot_number=SHOT_NUMBER)

    # Make BiLSTM predictions
    shot_data = make_predictions_on_shot(model, shot_data, selected_features, feature_scaler, device)

    # Classify with kNN cloud
    shot_data = classify_shot_with_knn_cloud(shot_data, imputer, knn_scaler, X_train_std, threshold)

    # Create visualizations
    fig1 = create_overlayed_visualization(shot_data)
    fig2 = create_detailed_analysis_plot(shot_data)

    # Print detailed statistics
    print_detailed_statistics(shot_data)

    # Create output directory if it doesn't exist
    output_dir = 'BiLSTM_Overlayed_Analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated directory: {output_dir}")

    # Save plots
    comprehensive_filename = os.path.join(output_dir, f'{OUTPUT_PREFIX}_comprehensive_overlayed.png')
    detailed_filename = os.path.join(output_dir, f'{OUTPUT_PREFIX}_detailed_analysis.png')

    fig1.savefig(comprehensive_filename, dpi=800, bbox_inches='tight')
    fig2.savefig(detailed_filename, dpi=800, bbox_inches='tight')
    print(f"\nPlots saved as:")
    print(f"  - {comprehensive_filename}")
    print(f"  - {detailed_filename}")

    return shot_data

if __name__ == "__main__":
    shot_data = main()
