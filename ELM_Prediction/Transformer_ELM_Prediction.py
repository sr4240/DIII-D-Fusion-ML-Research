# Auto-accept test - if you see this comment, auto-accept worked!
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
from collections import Counter
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
    Improved with Huber loss component for better gradient stability.
    """
    def __init__(self, max_horizon=150, weight_power=2.5, huber_delta=10.0):
        super(WeightedMSELoss, self).__init__()
        self.max_horizon = max_horizon
        self.weight_power = weight_power
        self.huber_delta = huber_delta
    
    def forward(self, predictions, targets):
        errors = predictions - targets
        abs_errors = torch.abs(errors)
        
        # Huber loss for better gradient behavior
        huber_loss = torch.where(
            abs_errors < self.huber_delta,
            0.5 * errors ** 2,
            self.huber_delta * (abs_errors - 0.5 * self.huber_delta)
        )
        
        # Weight by inverse of target (heavier weight for short-term)
        weights = torch.pow(self.max_horizon / (targets + 1.0), self.weight_power)
        weights = weights / weights.mean()
        weighted_loss = huber_loss * weights
        return weighted_loss.mean()


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer - adds temporal position information.
    """
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalTransformerRegressor(nn.Module):
    """
    Causal Transformer for ELM time prediction.
    Uses causal masking to prevent looking at future data.
    """
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.2, max_seq_len=150):
        super(CausalTransformerRegressor, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection: features -> d_model
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        
        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers for regression - improved architecture
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, 1)
        )
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"Causal Transformer Regressor - ELM Time Prediction")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Architecture: {num_layers}-layer Transformer (d_model={d_model}, heads={nhead})")
        print(f"Feedforward dim: {dim_feedforward}")
        print(f"NOTE: Causal masking - only sees past data, no future leakage")
        print(f"{'='*60}")
    
    def generate_causal_mask(self, seq_len, device):
        """Generate causal mask to prevent attending to future positions."""
        # True means masked (can't attend), False means can attend
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, x):
        # x: (batch_size, n_features, seq_len)
        batch_size, n_features, seq_len = x.shape
        
        # Transpose to (batch_size, seq_len, n_features)
        x = x.transpose(1, 2)
        
        # Project features to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate causal mask
        causal_mask = self.generate_causal_mask(seq_len, x.device)
        
        # Apply transformer encoder with causal mask
        x = self.transformer_encoder(x, mask=causal_mask)
        
        # Take the last position output (current time, after seeing all past)
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Output regression value
        output = self.output_layers(x)
        
        return output.squeeze(-1)  # (batch_size,)


class PlasmaRegressionDataset(Dataset):
    """Dataset class for plasma data windows with regression labels"""
    def __init__(self, windows, labels, sample_weights=None):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.FloatTensor(labels)
        self.sample_weights = torch.FloatTensor(sample_weights) if sample_weights is not None else None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if self.sample_weights is not None:
            return self.windows[idx].T, self.labels[idx], self.sample_weights[idx]
        return self.windows[idx].T, self.labels[idx]


def load_and_prepare_data():
    """Load and preprocess the plasma data"""
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    df = df[df['shot'] != 191675].copy()

    important_features = ['iln3iamp', 'betan', 'density', 'li',
                         'tritop', 'fs04_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    print(f"Using {len(selected_features)} features: {selected_features}")

    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()

    X = df_filtered[selected_features].values
    y = df_filtered['state'].values
    shots = df_filtered['shot'].values
    times = df_filtered['time'].values

    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    shots = shots[valid_mask]
    times = times[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"State distribution: {Counter(y)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, shots, times, selected_features, scaler


def compute_time_to_elm(states, current_idx, max_horizon=150):
    """Compute time until next ELM (state 4) from current position."""
    for i in range(1, max_horizon + 1):
        future_idx = current_idx + i
        if future_idx >= len(states):
            break
        if states[future_idx] == 4:
            return i
    return max_horizon


def create_causal_windows_with_temporal_split(X, y, shots, window_size=150, max_horizon=150):
    """Create causal windows and compute time-to-ELM labels with temporal split."""
    print(f"Creating causal windows of size {window_size}...")
    print(f"Prediction horizon: {max_horizon} timesteps")

    windows = []
    labels = []
    window_shots = []

    unique_shots = np.unique(shots)
    
    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        shot_states = y[shot_indices]

        for i in range(window_size - 1, len(shot_indices)):
            start_idx = shot_indices[i - window_size + 1]
            end_idx = shot_indices[i] + 1

            window = X[start_idx:end_idx]
            current_state = y[shot_indices[i]]

            if current_state == 4:
                continue

            current_pos_in_shot = i
            time_to_elm = compute_time_to_elm(shot_states, current_pos_in_shot, max_horizon)

            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(time_to_elm)
                window_shots.append(shot_id)

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    window_shots = np.array(window_shots)

    print(f"Created {len(windows)} valid causal windows")
    print(f"Label statistics: min={labels.min():.1f}, max={labels.max():.1f}, "
          f"mean={labels.mean():.1f}, std={labels.std():.1f}")

    # Temporal split
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

    train_mask = np.array([s in train_shots for s in window_shots])
    val_mask = np.array([s in val_shots for s in window_shots])
    test_mask = np.array([s in test_shots for s in window_shots])

    # Compute sample weights
    def compute_sample_weights(labels, max_horizon=150, weight_power=2.0):
        weights = np.power(max_horizon / (labels + 1.0), weight_power)
        weights = weights / weights.mean()
        return weights.astype(np.float32)
    
    train_weights = compute_sample_weights(labels[train_mask], max_horizon)
    val_weights = compute_sample_weights(labels[val_mask], max_horizon)
    test_weights = compute_sample_weights(labels[test_mask], max_horizon)
    
    print(f"\nSample weight statistics (train):")
    print(f"  Mean: {train_weights.mean():.3f}, Min: {train_weights.min():.3f}, Max: {train_weights.max():.3f}")
    
    return (windows[train_mask], labels[train_mask], train_weights,
            windows[val_mask], labels[val_mask], val_weights,
            windows[test_mask], labels[test_mask], test_weights)


def train_model(model, train_loader, val_loader, device, n_epochs=100, model_save_path=None, max_horizon=150):
    """Train the Transformer model with weighted loss"""
    criterion = WeightedMSELoss(max_horizon=max_horizon, weight_power=2.5, huber_delta=10.0)
    
    # Use AdamW optimizer with better hyperparameters
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.02, betas=(0.9, 0.999))
    
    # Improved warmup + cosine annealing scheduler
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            return 0.1 + 0.4 * (1 + math.cos(math.pi * progress))  # Minimum LR of 0.1 * initial
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    best_val_mae = float('inf')
    patience_counter = 0
    max_patience = 20  # Increased patience

    if model_save_path is None:
        model_save_path = os.path.join(SCRIPT_DIR, 'best_transformer_elm_predictor.pth')

    print("\nStarting training...")
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_data in train_loader:
            if len(batch_data) == 3:
                batch_X, batch_y, _ = batch_data
            else:
                batch_X, batch_y = batch_data
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Tighter gradient clipping
            optimizer.step()

            train_loss += loss.item()
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

        scheduler.step()

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
    """Evaluate the model on test set"""
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
    all_preds = np.clip(all_preds, 0, max_horizon)

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)

    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.2f} timesteps")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} timesteps")
    print(f"R-squared (R2): {r2:.4f}")

    tolerances = [5, 10, 20, 30]
    print("\nPrediction Accuracy within Tolerance:")
    for tol in tolerances:
        within_tol = np.mean(np.abs(all_preds - all_labels) <= tol) * 100
        print(f"  Within {tol:2d} timesteps: {within_tol:.1f}%")

    print("\nPerformance by Time-to-ELM Range:")
    ranges = [(0, 10), (10, 20), (20, 30), (30, 60), (60, 100), (100, 150)]
    for low, high in ranges:
        mask = (all_labels >= low) & (all_labels < high)
        if mask.sum() > 0:
            range_mae = mean_absolute_error(all_labels[mask], all_preds[mask])
            range_rmse = np.sqrt(mean_squared_error(all_labels[mask], all_preds[mask]))
            print(f"  {low:3d}-{high:3d} timesteps: MAE={range_mae:.2f}, RMSE={range_rmse:.2f}, n={mask.sum()}")
    
    short_term_mask = all_labels < 30
    if short_term_mask.sum() > 0:
        print(f"\nCRITICAL: Short-term predictions (<30 timesteps):")
        print(f"  Count: {short_term_mask.sum()}")
        print(f"  MAE: {mean_absolute_error(all_labels[short_term_mask], all_preds[short_term_mask]):.2f}")
        print(f"  RMSE: {np.sqrt(mean_squared_error(all_labels[short_term_mask], all_preds[short_term_mask])):.2f}")
        within_5 = np.mean(np.abs(all_preds[short_term_mask] - all_labels[short_term_mask]) <= 5) * 100
        within_10 = np.mean(np.abs(all_preds[short_term_mask] - all_labels[short_term_mask]) <= 10) * 100
        print(f"  Within 5 timesteps: {within_5:.1f}%")
        print(f"  Within 10 timesteps: {within_10:.1f}%")

    return all_preds, all_labels


def plot_results(train_losses, val_losses, train_maes, val_maes, all_preds, all_labels, max_horizon=150, save_path=None):
    """Plot training curves and regression diagnostics"""

    if save_path is None:
        save_path = os.path.join(SCRIPT_DIR, 'transformer_elm_prediction_results.png')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Training loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (Weighted MSE)')
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
    axes[0, 2].set_title('Predicted vs Actual Time to ELM')
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

    # Plot 6: Error vs Actual
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


def main():
    """Main training pipeline for Transformer ELM time prediction"""
    print("=" * 60)
    print("Causal Transformer Model for ELM Time Prediction")
    print("=" * 60)
    print("Task: Predict time (timesteps) until next ELM event")
    print("Model: Transformer with causal masking (no future data leakage)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    window_size = 150
    max_horizon = 150
    batch_size = 256

    # Paths
    model_save_path = os.path.join(SCRIPT_DIR, 'best_transformer_elm_predictor.pth')
    results_save_path = os.path.join(SCRIPT_DIR, 'transformer_elm_prediction_results.png')
    print(f"\nOutput directory: {SCRIPT_DIR}")
    print(f"  Model will be saved to: {model_save_path}")
    print(f"  Results plot will be saved to: {results_save_path}")

    # Load data
    X, y, shots, times, features, scaler = load_and_prepare_data()

    # Create causal windows
    train_X, train_y, train_weights, val_X, val_y, val_weights, test_X, test_y, test_weights = create_causal_windows_with_temporal_split(
        X, y, shots, window_size=window_size, max_horizon=max_horizon
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")
    print(f"  Batch size: {batch_size}")

    # Create data loaders
    train_dataset = PlasmaRegressionDataset(train_X, train_y, train_weights)
    val_dataset = PlasmaRegressionDataset(val_X, val_y, val_weights)
    test_dataset = PlasmaRegressionDataset(test_X, test_y, test_weights)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Create Transformer model with improved architecture
    model = CausalTransformerRegressor(
        n_features=len(features),
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.15,  # Reduced dropout for better learning
        max_seq_len=window_size
    ).to(device)

    # Train model
    print("\nStarting training with Transformer architecture...")
    train_losses, val_losses, train_maes, val_maes = train_model(
        model, train_loader, val_loader, device, n_epochs=100, model_save_path=model_save_path, max_horizon=max_horizon
    )

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load(model_save_path))

    # Evaluate
    all_preds, all_labels = evaluate_model(model, test_loader, device, max_horizon)

    # Plot results
    plot_results(train_losses, val_losses, train_maes, val_maes, all_preds, all_labels, max_horizon, results_save_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
