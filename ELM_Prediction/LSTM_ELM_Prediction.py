import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that heavily penalizes errors in short-term predictions.
    Weight increases exponentially as actual time-to-ELM decreases.
    """
    def __init__(self, max_horizon=150, weight_power=2.0):
        super(WeightedMSELoss, self).__init__()
        self.max_horizon = max_horizon
        self.weight_power = weight_power
    
    def forward(self, predictions, targets):
        # Compute base MSE
        mse = (predictions - targets) ** 2
        
        # Compute weights: higher weight for shorter times-to-ELM
        # Weight = (max_horizon / (target + 1))^power
        # This gives much higher weight to short-term predictions
        weights = torch.pow(self.max_horizon / (targets + 1.0), self.weight_power)
        
        # Normalize weights to have mean of 1 (keeps loss scale similar)
        weights = weights / weights.mean()
        
        weighted_mse = mse * weights
        return weighted_mse.mean()


class CausalLSTMRegressor(nn.Module):
    """
    Improved Causal (unidirectional) LSTM for ELM time prediction.
    Only uses past data - no future leakage.
    Enhanced architecture for better short-term prediction.
    """
    def __init__(self, n_features, lstm_hidden=256, nn_hidden_sizes=[512, 256, 128]):
        super(CausalLSTMRegressor, self).__init__()

        # Unidirectional LSTM - only processes past -> present (NO bidirectional!)
        # Increased hidden size and layers for better capacity
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=3,  # Deeper LSTM
            batch_first=True,
            bidirectional=False,  # CRITICAL: Causal, forward-only
            dropout=0.3
        )

        # Output size is just lstm_hidden (not *2 since no bidirectional)
        lstm_output_size = lstm_hidden

        # NN layers process the LSTM output - deeper network
        nn_layers = []
        input_dim = lstm_output_size

        for hidden_size in nn_hidden_sizes:
            nn_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_size

        self.nn_layers = nn.Sequential(*nn_layers)

        # Attention weights for temporal aggregation
        self.attention_weights = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Final regressor (single output for time prediction) - deeper
        self.regressor = nn.Sequential(
            nn.Linear(input_dim + lstm_output_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Single output: time to next ELM
        )

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"Improved Causal LSTM Regressor - ELM Time Prediction")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Architecture: 3-layer LSTM ({lstm_hidden} hidden) -> Deep NN -> Regressor")
        print(f"Improvements: Weighted loss, deeper network, better regularization")
        print(f"NOTE: Causal model - only sees past data, no future leakage")
        print(f"{'='*60}")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        batch_size, n_features, seq_len = x.shape

        # Transpose for LSTM: (batch_size, sequence_length, n_features)
        x = x.transpose(1, 2)

        # LSTM processes the temporal sequence (forward only)
        lstm_output, (hidden, cell) = self.lstm(x)
        # lstm_output shape: (batch_size, seq_len, lstm_hidden)

        # Apply attention to aggregate temporal information
        attention = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        attended_features = torch.sum(lstm_output * attention, dim=1)  # (batch_size, lstm_hidden)

        # Take the final hidden state (represents "current" time after seeing all past)
        final_hidden = lstm_output[:, -1, :]  # (batch_size, lstm_hidden)

        # Process through NN layers
        nn_features = self.nn_layers(final_hidden)

        # Combine attended features with NN features
        combined = torch.cat([nn_features, attended_features], dim=1)

        # Regression output
        output = self.regressor(combined)

        return output.squeeze(-1)  # (batch_size,)


class PlasmaRegressionDataset(Dataset):
    """Dataset class for plasma data windows with regression labels"""
    def __init__(self, windows, labels, weights=None):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.FloatTensor(labels)  # Float for regression
        self.weights = torch.FloatTensor(weights) if weights is not None else None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Transpose to get (n_features, sequence_length) format
        if self.weights is not None:
            return self.windows[idx].T, self.labels[idx], self.weights[idx]
        return self.windows[idx].T, self.labels[idx]


def load_and_prepare_data():
    """Load and preprocess the plasma data"""
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    # Select only the specified features
    important_features = ['iln3iamp', 'betan', 'density', 'li',
                         'tritop', 'fs04_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    print(f"Using {len(selected_features)} features: {selected_features}")

    # Sort by shot and time
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Filter out state 0 (keep states 1-4)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()

    # Extract features and labels
    X = df_filtered[selected_features].values
    y = df_filtered['state'].values
    shots = df_filtered['shot'].values
    times = df_filtered['time'].values

    # Remove NaN values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    shots = shots[valid_mask]
    times = times[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"State distribution: {Counter(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, shots, times, selected_features, scaler


def compute_time_to_elm(states, current_idx, max_horizon=150):
    """
    Compute time (in timesteps) until next ELM (state 4) from current position.
    
    Args:
        states: Array of states for the shot
        current_idx: Current position in the shot (relative to shot start)
        max_horizon: Maximum prediction horizon (cap value)
    
    Returns:
        Time to next ELM, or max_horizon if no ELM within horizon
    """
    # Look forward from current position
    for i in range(1, max_horizon + 1):
        future_idx = current_idx + i
        if future_idx >= len(states):
            break
        if states[future_idx] == 4:  # Found ELM
            return i
    
    return max_horizon  # No ELM found within horizon


def create_causal_windows_with_temporal_split(X, y, shots, window_size=150, max_horizon=150, suppressed_state=1):
    """
    Create causal windows (look-back only) and compute time-to-ELM labels.
    Uses temporal shot-based split to prevent data leakage.
    
    FOCUS: Only considers data around suppressed state (state 1) to ELMing state (state 4) transitions.
    The transition point is the disruption point we want to predict.
    """
    print(f"Creating causal windows of size {window_size}...")
    print(f"Prediction horizon: {max_horizon} timesteps")
    print(f"FOCUS: Only considering suppressed state ({suppressed_state}) → ELMing (4) transitions")

    windows = []
    labels = []
    window_shots = []  # Track which shot each window belongs to

    # Process each shot
    unique_shots = np.unique(shots)
    
    skipped_not_suppressed = 0
    skipped_no_elm = 0
    
    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        # Get states for this shot
        shot_states = y[shot_indices]

        # Create causal windows: window ends at current time (no future data)
        for i in range(window_size - 1, len(shot_indices)):
            # Window: from (i - window_size + 1) to i (inclusive)
            start_idx = shot_indices[i - window_size + 1]
            end_idx = shot_indices[i] + 1  # +1 for slice

            window = X[start_idx:end_idx]

            # Current state (at end of window)
            current_state = y[shot_indices[i]]

            # ONLY consider suppressed state (state 1) - focus on suppressed → ELMing transitions
            if current_state != suppressed_state:
                skipped_not_suppressed += 1
                continue

            # Compute time to next ELM from current position
            current_pos_in_shot = i
            time_to_elm = compute_time_to_elm(shot_states, current_pos_in_shot, max_horizon)
            
            # Skip if no actual ELM event within horizon (we only want real transitions)
            if time_to_elm >= max_horizon:
                skipped_no_elm += 1
                continue

            # Check window validity
            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(time_to_elm)
                window_shots.append(shot_id)

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    window_shots = np.array(window_shots)

    print(f"\nCreated {len(windows)} valid causal windows (suppressed → ELMing only)")
    print(f"Skipped {skipped_not_suppressed} windows (not in suppressed state)")
    print(f"Skipped {skipped_no_elm} windows (no ELM within horizon)")
    print(f"Label statistics: min={labels.min():.1f}, max={labels.max():.1f}, "
          f"mean={labels.mean():.1f}, std={labels.std():.1f}")

    # Temporal split by shots (chronological)
    unique_shots_sorted = np.sort(np.unique(window_shots))
    n_shots = len(unique_shots_sorted)

    train_shot_end = int(0.7 * n_shots)
    val_shot_end = int(0.85 * n_shots)

    train_shots = set(unique_shots_sorted[:train_shot_end])
    val_shots = set(unique_shots_sorted[train_shot_end:val_shot_end])
    test_shots = set(unique_shots_sorted[val_shot_end:])

    print(f"\nTemporal split:")
    print(f"  Train shots: {len(train_shots)} ({unique_shots_sorted[0]} to {unique_shots_sorted[train_shot_end-1]})")
    print(f"  Val shots: {len(val_shots)} ({unique_shots_sorted[train_shot_end]} to {unique_shots_sorted[val_shot_end-1]})")
    print(f"  Test shots: {len(test_shots)} ({unique_shots_sorted[val_shot_end]} to {unique_shots_sorted[-1]})")

    # Create masks
    train_mask = np.array([s in train_shots for s in window_shots])
    val_mask = np.array([s in val_shots for s in window_shots])
    test_mask = np.array([s in test_shots for s in window_shots])

    # Compute sample weights for training (heavier weight on short-term predictions)
    def compute_sample_weights(labels, max_horizon=150, weight_power=2.0):
        """Compute weights: higher weight for shorter time-to-ELM"""
        weights = np.power(max_horizon / (labels + 1.0), weight_power)
        # Normalize to have mean of 1
        weights = weights / weights.mean()
        return weights.astype(np.float32)
    
    train_weights = compute_sample_weights(labels[train_mask], max_horizon)
    val_weights = compute_sample_weights(labels[val_mask], max_horizon)
    test_weights = compute_sample_weights(labels[test_mask], max_horizon)
    
    print(f"\nSample weight statistics (train):")
    print(f"  Mean: {train_weights.mean():.3f}, Min: {train_weights.min():.3f}, Max: {train_weights.max():.3f}")
    print(f"  Weight for 1 timestep: {compute_sample_weights(np.array([1]), max_horizon)[0]:.2f}")
    print(f"  Weight for 150 timesteps: {compute_sample_weights(np.array([150]), max_horizon)[0]:.2f}")
    
    return (windows[train_mask], labels[train_mask], train_weights,
            windows[val_mask], labels[val_mask], val_weights,
            windows[test_mask], labels[test_mask], test_weights)


def train_model(model, train_loader, val_loader, device, n_epochs=100, model_save_path=None, max_horizon=150):
    """Train the regression model with weighted loss and cosine annealing"""
    # Use weighted MSE loss with higher weight_power for more focus on short-term
    criterion = WeightedMSELoss(max_horizon=max_horizon, weight_power=3.0)
    
    # AdamW optimizer (better for regularization)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Cosine annealing with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training history
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    best_val_mae = float('inf')
    patience_counter = 0
    max_patience = 15  # Increased patience

    if model_save_path is None:
        model_save_path = os.path.join(SCRIPT_DIR, 'best_lstm_elm_predictor.pth')

    print("\nStarting training...")
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                batch_X, batch_y, batch_weights = batch_data
                batch_X, batch_y, batch_weights = batch_X.to(device), batch_y.to(device), batch_weights.to(device)
            else:
                batch_X, batch_y = batch_data
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_weights = None

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            # Store predictions
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    batch_X, batch_y, _ = batch_data
                else:
                    batch_X, batch_y = batch_data
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()

                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(batch_y.cpu().numpy())

        # Calculate metrics
        train_mae = mean_absolute_error(train_labels, train_preds)
        val_mae = mean_absolute_error(val_labels_list, val_preds)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{n_epochs} (LR: {current_lr:.6f})")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {train_mae:.2f} timesteps")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.2f} timesteps")

        # Step the scheduler (cosine annealing)
        scheduler.step()

        # Early stopping based on MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print(f"  New best model saved! (MAE: {val_mae:.2f})")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_maes, val_maes


def evaluate_model(model, test_loader, device, max_horizon=150):
    """Evaluate the regression model on test set"""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                batch_X, batch_y, _ = batch_data
            else:
                batch_X, batch_y = batch_data
            
            batch_X = batch_X.to(device)

            outputs = model(batch_X)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Clip predictions to valid range
    all_preds = np.clip(all_preds, 0, max_horizon)

    # Calculate metrics
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)

    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.2f} timesteps")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} timesteps")
    print(f"R-squared (R2): {r2:.4f}")

    # Tolerance-based accuracy
    tolerances = [5, 10, 20, 30]
    print("\nPrediction Accuracy within Tolerance:")
    for tol in tolerances:
        within_tol = np.mean(np.abs(all_preds - all_labels) <= tol) * 100
        print(f"  Within {tol:2d} timesteps: {within_tol:.1f}%")

    # Breakdown by label range
    print("\nPerformance by Time-to-ELM Range:")
    ranges = [(0, 30), (30, 60), (60, 100), (100, 150)]
    for low, high in ranges:
        mask = (all_labels >= low) & (all_labels < high)
        if mask.sum() > 0:
            range_mae = mean_absolute_error(all_labels[mask], all_preds[mask])
            print(f"  {low:3d}-{high:3d} timesteps: MAE={range_mae:.2f}, n={mask.sum()}")

    return all_preds, all_labels


def plot_results(train_losses, val_losses, train_maes, val_maes, all_preds, all_labels, max_horizon=150, save_path=None):
    """Plot training curves and regression diagnostics"""

    if save_path is None:
        save_path = os.path.join(SCRIPT_DIR, 'lstm_elm_prediction_results.png')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Training loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (SmoothL1)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: MAE over epochs
    axes[0, 1].plot(train_maes, label='Train MAE', color='blue')
    axes[0, 1].plot(val_maes, label='Val MAE', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (timesteps)')
    axes[0, 1].set_title('Mean Absolute Error over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Predicted vs Actual scatter
    axes[0, 2].scatter(all_labels, all_preds, alpha=0.3, s=10)
    axes[0, 2].plot([0, max_horizon], [0, max_horizon], 'r--', linewidth=2, label='Perfect prediction')
    axes[0, 2].set_xlabel('Actual Time to ELM (timesteps)')
    axes[0, 2].set_ylabel('Predicted Time to ELM (timesteps)')
    axes[0, 2].set_title('Predicted vs Actual (Suppressed → ELMing)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(0, max_horizon)
    axes[0, 2].set_ylim(0, max_horizon)

    # Plot 4: Error distribution
    errors = all_preds - all_labels
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.1f}')
    axes[1, 0].set_xlabel('Prediction Error (timesteps)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Absolute error distribution
    abs_errors = np.abs(errors)
    axes[1, 1].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].axvline(x=np.mean(abs_errors), color='red', linestyle='--', linewidth=2, 
                       label=f'MAE: {np.mean(abs_errors):.1f}')
    axes[1, 1].axvline(x=np.median(abs_errors), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(abs_errors):.1f}')
    axes[1, 1].set_xlabel('Absolute Error (timesteps)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Absolute Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Error vs Actual (to check for systematic bias)
    axes[1, 2].scatter(all_labels, errors, alpha=0.3, s=10)
    axes[1, 2].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 2].set_xlabel('Actual Time to ELM (timesteps)')
    axes[1, 2].set_ylabel('Prediction Error (timesteps)')
    axes[1, 2].set_title('Error vs Actual Value')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nResults saved to '{save_path}'")


def visualize_random_shots(model, X, y, shots, scaler, features, device, window_size=150, max_horizon=150, n_shots=3, save_path=None, suppressed_state=1):
    """
    Visualize model predictions on 3 random shots to convey performance visually.
    Shows actual vs predicted time-to-ELM over the course of each shot.
    
    FOCUS: Only considers suppressed state (state 1) → ELMing (state 4) transitions.
    """
    if save_path is None:
        save_path = os.path.join(SCRIPT_DIR, 'lstm_random_shots_visualization.png')
    
    model.eval()
    
    # Get unique shots and select 3 random ones
    # Filter shots that have enough suppressed state data points AND actual ELM events
    unique_shots = np.unique(shots)
    valid_shots = []
    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_states = y[shot_mask]
        # Need enough data AND both suppressed and ELMing states present
        has_suppressed = np.sum(shot_states == suppressed_state) >= window_size
        has_elm = np.sum(shot_states == 4) > 0
        if has_suppressed and has_elm:
            valid_shots.append(shot_id)
    
    np.random.seed(None)  # Use random seed for variety
    selected_shots = np.random.choice(valid_shots, size=min(n_shots, len(valid_shots)), replace=False)
    
    print(f"\n{'='*60}")
    print(f"Visualizing Suppressed → ELMing Transitions on {len(selected_shots)} Random Shots")
    print(f"Selected shots: {selected_shots}")
    print(f"{'='*60}")
    
    # Create figure with subplots for each shot
    fig, axes = plt.subplots(n_shots, 2, figsize=(16, 5*n_shots))
    if n_shots == 1:
        axes = axes.reshape(1, -1)
    
    # Color scheme
    colors = {
        'actual': '#2E86AB',      # Deep blue
        'predicted': '#E94F37',    # Coral red
        'elm_marker': '#F39C12',   # Orange
        'error_fill': '#95A5A6',   # Gray
    }
    
    for idx, shot_id in enumerate(selected_shots):
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]
        shot_states = y[shot_indices]
        shot_X = X[shot_indices]
        
        # Compute predictions for each valid window position
        predictions = []
        actual_times = []
        positions = []
        
        for i in range(window_size - 1, len(shot_indices)):
            # ONLY consider suppressed state → ELMing transitions
            current_state = shot_states[i]
            if current_state != suppressed_state:
                continue
            
            # Create window
            start_idx = i - window_size + 1
            window = shot_X[start_idx:i+1]
            
            if np.isnan(window).any() or np.isinf(window).any():
                continue
            
            # Compute actual time to ELM
            time_to_elm = compute_time_to_elm(shot_states, i, max_horizon)
            
            # Skip if no actual ELM within horizon (we only want real transitions)
            if time_to_elm >= max_horizon:
                continue
            
            # Get prediction
            window_tensor = torch.FloatTensor(window).unsqueeze(0).transpose(1, 2).to(device)
            with torch.no_grad():
                pred = model(window_tensor).cpu().numpy()[0]
            pred = np.clip(pred, 0, max_horizon)
            
            predictions.append(pred)
            actual_times.append(time_to_elm)
            positions.append(i - window_size + 1)  # Relative position in shot
        
        predictions = np.array(predictions)
        actual_times = np.array(actual_times)
        positions = np.array(positions)
        
        # Find ELM events (state 4) in the shot
        elm_positions = np.where(shot_states == 4)[0] - window_size + 1
        elm_positions = elm_positions[elm_positions >= 0]
        
        # Skip if no valid data points for this shot
        if len(predictions) == 0:
            print(f"\nShot {shot_id}: No valid Suppressed→ELMing transitions found, skipping...")
            axes[idx, 0].text(0.5, 0.5, f'Shot {shot_id}\nNo valid transitions found', 
                             ha='center', va='center', fontsize=12, transform=axes[idx, 0].transAxes)
            axes[idx, 0].set_title(f'Shot {shot_id} - No Data', fontsize=12)
            axes[idx, 1].text(0.5, 0.5, f'Shot {shot_id}\nNo valid transitions found', 
                             ha='center', va='center', fontsize=12, transform=axes[idx, 1].transAxes)
            axes[idx, 1].set_title(f'Shot {shot_id} - No Data', fontsize=12)
            continue
        
        # Plot 1: Actual vs Predicted over time
        ax1 = axes[idx, 0]
        ax1.plot(positions, actual_times, color=colors['actual'], linewidth=2, 
                 label='Actual Time to ELM', alpha=0.9)
        ax1.plot(positions, predictions, color=colors['predicted'], linewidth=2, 
                 label='Predicted Time to ELM', alpha=0.9)
        
        # Mark ELM events
        for elm_pos in elm_positions:
            ax1.axvline(x=elm_pos, color=colors['elm_marker'], linestyle='--', 
                       alpha=0.7, linewidth=1.5)
        
        # Add error shading
        ax1.fill_between(positions, actual_times, predictions, 
                        color=colors['error_fill'], alpha=0.3, label='Error')
        
        ax1.set_xlabel('Time Step in Shot (Suppressed State Only)', fontsize=11)
        ax1.set_ylabel('Time to ELM (timesteps)', fontsize=11)
        ax1.set_title(f'Shot {shot_id} - Suppressed → ELMing Transition', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max_horizon + 10)
        
        # Calculate metrics for this shot
        shot_mae = mean_absolute_error(actual_times, predictions)
        shot_rmse = np.sqrt(mean_squared_error(actual_times, predictions))
        
        # Add text box with metrics
        textstr = f'MAE: {shot_mae:.1f}\nRMSE: {shot_rmse:.1f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Plot 2: Scatter plot with error analysis
        ax2 = axes[idx, 1]
        
        # Create scatter with color based on actual time (short-term in one color, long-term in another)
        scatter = ax2.scatter(actual_times, predictions, c=actual_times, cmap='viridis', 
                             alpha=0.6, s=30, edgecolors='none')
        
        # Perfect prediction line
        ax2.plot([0, max_horizon], [0, max_horizon], 'r--', linewidth=2, 
                label='Perfect Prediction', alpha=0.8)
        
        # Tolerance bands
        ax2.fill_between([0, max_horizon], [0-10, max_horizon-10], [0+10, max_horizon+10],
                        color='green', alpha=0.1, label='±10 timesteps')
        
        ax2.set_xlabel('Actual Time to ELM (timesteps)', fontsize=11)
        ax2.set_ylabel('Predicted Time to ELM (timesteps)', fontsize=11)
        ax2.set_title(f'Shot {shot_id} - Prediction Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max_horizon)
        ax2.set_ylim(0, max_horizon)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Actual Time to ELM', fontsize=10)
        
        # Calculate accuracy within tolerances
        within_10 = np.mean(np.abs(predictions - actual_times) <= 10) * 100
        within_20 = np.mean(np.abs(predictions - actual_times) <= 20) * 100
        
        textstr2 = f'±10: {within_10:.0f}%\n±20: {within_20:.0f}%'
        ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        print(f"\nShot {shot_id}:")
        print(f"  MAE: {shot_mae:.2f} timesteps")
        print(f"  RMSE: {shot_rmse:.2f} timesteps")
        print(f"  Within ±10: {within_10:.1f}%")
        print(f"  Within ±20: {within_20:.1f}%")
        print(f"  Data points: {len(predictions)}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nRandom shots visualization saved to '{save_path}'")


def main():
    """Main training pipeline for ELM time prediction"""
    print("=" * 60)
    print("Causal LSTM Model for ELM Time Prediction")
    print("=" * 60)
    print("Task: Predict time (timesteps) until next ELM event")
    print("FOCUS: Only suppressed state (1) → ELMing (4) transitions")
    print("       (Disruption point = transition from suppressed to ELMing)")
    print("Model: Unidirectional LSTM (causal - no future data leakage)")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    window_size = 150
    max_horizon = 150

    # Paths for saving outputs (all saved to script directory)
    model_save_path = os.path.join(SCRIPT_DIR, 'best_lstm_elm_predictor.pth')
    results_save_path = os.path.join(SCRIPT_DIR, 'lstm_elm_prediction_results.png')
    print(f"\nOutput directory: {SCRIPT_DIR}")
    print(f"  Model will be saved to: {model_save_path}")
    print(f"  Results plot will be saved to: {results_save_path}")

    # Load data
    X, y, shots, times, features, scaler = load_and_prepare_data()

    # Create causal windows with temporal split (now returns weights too)
    train_X, train_y, train_weights, val_X, val_y, val_weights, test_X, test_y, test_weights = create_causal_windows_with_temporal_split(
        X, y, shots, window_size=window_size, max_horizon=max_horizon
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    # Create data loaders with sample weights
    # Batch size 256: Better GPU utilization, faster training
    batch_size = 256
    train_dataset = PlasmaRegressionDataset(train_X, train_y, train_weights)
    val_dataset = PlasmaRegressionDataset(val_X, val_y, val_weights)
    test_dataset = PlasmaRegressionDataset(test_X, test_y, test_weights)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Batch size: {batch_size}")

    # Create model (improved architecture)
    model = CausalLSTMRegressor(n_features=len(features), lstm_hidden=256, nn_hidden_sizes=[512, 256, 128]).to(device)

    # Train model
    print("\nStarting training with weighted loss (focusing on short-term predictions)...")
    train_losses, val_losses, train_maes, val_maes = train_model(
        model, train_loader, val_loader, device, n_epochs=100, model_save_path=model_save_path, max_horizon=max_horizon
    )

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load(model_save_path))

    # Evaluate on test set
    all_preds, all_labels = evaluate_model(model, test_loader, device, max_horizon)

    # Plot results
    plot_results(train_losses, val_losses, train_maes, val_maes, all_preds, all_labels, max_horizon, results_save_path)

    # Visualize 3 random shots to show performance
    random_shots_save_path = os.path.join(SCRIPT_DIR, 'lstm_random_shots_visualization.png')
    visualize_random_shots(
        model, X, y, shots, scaler, features, device,
        window_size=window_size, max_horizon=max_horizon, n_shots=3,
        save_path=random_shots_save_path
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
