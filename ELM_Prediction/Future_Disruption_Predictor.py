"""
Future Disruption Predictor
============================
Predicts whether a transition OUT of Suppressed state will occur
within various future time horizons using a Causal LSTM Binary Classifier.

Key Features:
- Look-back window: Up to 500 ms (500 timesteps at 1ms/sample)
- Disruption states: State 2 (Dithering), State 3 (Mitigated), State 4 (ELMing)
- Suppressed state: State 1 (the "safe" state)
- Prediction horizons: 30, 50, 80, 100, 120, 150, 200 ms
- Output: Binary (1 = transition from Suppressed to disruption within horizon, 0 = stays Suppressed)
- Only considers samples currently in Suppressed state (State 1)
- Separate models trained for each horizon
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             classification_report, balanced_accuracy_score)
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

# Prediction horizons in milliseconds (= timesteps at 1ms sampling)
HORIZONS = [30, 50, 80, 100, 120, 150, 200]

# Suppressed state (the "safe" state we want to stay in)
SUPPRESSED_STATE = 1

# Disruption states: State 2 (Dithering), State 3 (Mitigated), State 4 (ELMing)
# These represent transitions OUT of the Suppressed state
DISRUPTION_STATES = [2, 3, 4]


class CausalLSTMBinaryClassifier(nn.Module):
    """
    Causal (unidirectional) LSTM for binary disruption prediction.
    Only uses past data - no future leakage.
    Outputs probability of disruption within the specified horizon.
    """
    def __init__(self, n_features, lstm_hidden=256, nn_hidden_sizes=[512, 256, 128]):
        super(CausalLSTMBinaryClassifier, self).__init__()

        # Unidirectional LSTM - only processes past -> present (NO bidirectional!)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=False,  # CRITICAL: Causal, forward-only
            dropout=0.3
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

        # Final binary classifier (single output with sigmoid for probability)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim + lstm_output_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Single output for binary classification
        )

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"Causal LSTM Binary Classifier - Disruption Prediction")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Architecture: 3-layer LSTM ({lstm_hidden} hidden) -> NN -> Binary Classifier")
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

        # Take the final hidden state
        final_hidden = lstm_output[:, -1, :]  # (batch_size, lstm_hidden)

        # Process through NN layers
        nn_features = self.nn_layers(final_hidden)

        # Combine attended features with NN features
        combined = torch.cat([nn_features, attended_features], dim=1)

        # Binary classification output (logits)
        output = self.classifier(combined)

        return output.squeeze(-1)  # (batch_size,)


class PlasmaClassificationDataset(Dataset):
    """Dataset class for plasma data windows with binary labels"""
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.FloatTensor(labels)

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

    # Select features (same as existing ELM prediction)
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


def check_transition_in_horizon(states, current_idx, horizon):
    """
    Check if a transition from Suppressed (State 1) to a disruption state 
    (State 2, 3, or 4) occurs within the specified horizon.
    
    This detects the TRANSITION POINT - when plasma leaves the Suppressed state.
    
    Args:
        states: Array of states for the shot
        current_idx: Current position in the shot (relative to shot start)
        horizon: Number of timesteps to look ahead
    
    Returns:
        1 if transition to disruption found within horizon, 0 otherwise
    """
    # Current state must be Suppressed (1) for us to be looking for a transition
    if states[current_idx] != SUPPRESSED_STATE:
        return -1  # Invalid: not currently in Suppressed state
    
    for i in range(1, horizon + 1):
        future_idx = current_idx + i
        if future_idx >= len(states):
            break
        # Check if we transition to a disruption state
        if states[future_idx] in DISRUPTION_STATES:
            return 1  # Transition detected
    
    return 0  # Stays in Suppressed state within horizon


def create_windows_for_horizon(X, y, shots, window_size=500, horizon=100):
    """
    Create causal windows (look-back only) with binary disruption labels
    for a specific prediction horizon.
    
    ONLY includes samples where current state is Suppressed (State 1).
    Predicts transition from Suppressed to any disruption state (2, 3, or 4).
    
    Uses temporal shot-based split to prevent data leakage.
    """
    print(f"\nCreating windows for {horizon}ms horizon...")
    print(f"  Window size (lookback): {window_size} timesteps")
    print(f"  Only including samples currently in Suppressed state (State 1)")

    windows = []
    labels = []
    window_shots = []

    unique_shots = np.unique(shots)
    
    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        # Use smaller window if shot is shorter than window_size
        effective_window = min(window_size, len(shot_indices))
        
        if effective_window < 50:  # Minimum window size
            continue

        # Get states for this shot
        shot_states = y[shot_indices]

        # Create causal windows: window ends at current time (no future data)
        for i in range(effective_window - 1, len(shot_indices) - horizon):
            # Window: from (i - effective_window + 1) to i (inclusive)
            start_idx = shot_indices[i - effective_window + 1]
            end_idx = shot_indices[i] + 1

            window = X[start_idx:end_idx]

            # Current state (at end of window)
            current_state = y[shot_indices[i]]

            # ONLY include samples where current state is Suppressed (State 1)
            # We're predicting transition OUT of Suppressed state
            if current_state != SUPPRESSED_STATE:
                continue

            # Check if transition to disruption occurs within horizon
            current_pos_in_shot = i
            label = check_transition_in_horizon(shot_states, current_pos_in_shot, horizon)
            
            # Skip invalid labels (shouldn't happen now, but safety check)
            if label == -1:
                continue

            # Pad window to window_size if needed
            if len(window) < window_size:
                padding = np.zeros((window_size - len(window), window.shape[1]))
                window = np.vstack([padding, window])

            # Check window validity
            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(label)
                window_shots.append(shot_id)

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    window_shots = np.array(window_shots)

    print(f"  Created {len(windows)} valid windows (all from Suppressed state)")
    print(f"  Label distribution: {Counter(labels)}")
    print(f"    0 = Stays Suppressed, 1 = Transitions to Dithering/Mitigated/ELMing")
    positive_rate = labels.mean() * 100
    print(f"  Transition rate (leaves Suppressed within horizon): {positive_rate:.1f}%")

    # Temporal split by shots (chronological)
    unique_shots_sorted = np.sort(np.unique(window_shots))
    n_shots = len(unique_shots_sorted)

    train_shot_end = int(0.7 * n_shots)
    val_shot_end = int(0.85 * n_shots)

    train_shots = set(unique_shots_sorted[:train_shot_end])
    val_shots = set(unique_shots_sorted[train_shot_end:val_shot_end])
    test_shots = set(unique_shots_sorted[val_shot_end:])

    # Create masks
    train_mask = np.array([s in train_shots for s in window_shots])
    val_mask = np.array([s in val_shots for s in window_shots])
    test_mask = np.array([s in test_shots for s in window_shots])

    return (windows[train_mask], labels[train_mask],
            windows[val_mask], labels[val_mask],
            windows[test_mask], labels[test_mask])


class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    Reduces loss for well-classified examples, focusing on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing parameter
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # For numerical stability
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        
        # Focal weight: (1 - p_t)^gamma
        # p_t = p for positive class, 1-p for negative class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Binary cross-entropy
        bce = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)
        
        loss = alpha_t * focal_weight * bce
        return loss.mean()


def train_model_for_horizon(model, train_loader, val_loader, device, horizon, 
                            n_epochs=80, model_save_path=None, pos_weight=None):
    """Train the binary classification model for a specific horizon
    
    Args:
        pos_weight: Weight for positive class to handle class imbalance.
                   Calculated as (num_negatives / num_positives).
    """
    
    # Use Focal Loss for extreme class imbalance (better than weighted BCE)
    # Alpha balances positive/negative, gamma focuses on hard examples
    # Higher alpha = more weight on positive class
    alpha = min(0.75, 1.0 / (1.0 + 1.0/pos_weight)) if pos_weight else 0.25
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    print(f"  Using Focal Loss with alpha={alpha:.3f}, gamma=2.0")
    
    # AdamW optimizer with lower learning rate for stability
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # Cosine annealing scheduler
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    best_val_f1 = 0.0  # Use F1 score for early stopping (better for imbalanced data)
    patience_counter = 0
    max_patience = 20  # More patience for learning
    min_epochs = 30    # Don't stop before this many epochs

    if model_save_path is None:
        model_save_path = os.path.join(SCRIPT_DIR, f'disruption_predictor_{horizon}ms.pth')

    print(f"\nTraining model for {horizon}ms horizon...")
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            # Store predictions (convert logits to predictions)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_probs, val_labels_list = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                val_probs.extend(probs.cpu().numpy())
                val_labels_list.extend(batch_y.cpu().numpy())

        val_probs = np.array(val_probs)
        val_labels_arr = np.array(val_labels_list)
        
        # Find optimal threshold that maximizes F1 on validation set
        best_threshold = 0.5
        best_f1_for_threshold = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds_at_thresh = (val_probs >= thresh).astype(float)
            f1_at_thresh = f1_score(val_labels_arr, preds_at_thresh, zero_division=0)
            if f1_at_thresh > best_f1_for_threshold:
                best_f1_for_threshold = f1_at_thresh
                best_threshold = thresh
        
        # Use optimal threshold for predictions
        val_preds = (val_probs >= best_threshold).astype(float)
        train_preds_arr = np.array(train_preds)

        # Calculate metrics
        train_acc = accuracy_score(train_labels, train_preds_arr)
        val_acc = accuracy_score(val_labels_arr, val_preds)
        train_f1 = f1_score(train_labels, train_preds_arr, zero_division=0)
        val_f1 = best_f1_for_threshold  # Use the best F1 found with optimal threshold

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Val F1={val_f1:.4f} (thresh={best_threshold:.2f}), Val Acc={val_acc:.4f}")

        scheduler.step()

        # Early stopping based on validation F1 score (better for imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Only allow early stopping after min_epochs
        if epoch >= min_epochs and patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Best validation F1 score: {best_val_f1:.4f}")
    return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, best_val_f1


def find_optimal_threshold(probs, labels):
    """Find optimal classification threshold that maximizes F1 score"""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= threshold).astype(float)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def evaluate_model_for_horizon(model, test_loader, device, horizon, val_loader=None):
    """Evaluate the model on test set and return comprehensive metrics"""
    model.eval()

    # First, find optimal threshold on validation set if provided
    if val_loader is not None:
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.sigmoid(outputs)
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_y.numpy())
        
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        optimal_threshold, val_f1 = find_optimal_threshold(val_probs, val_labels)
        print(f"  Optimal threshold from validation: {optimal_threshold:.2f} (Val F1: {val_f1:.4f})")
    else:
        optimal_threshold = 0.5

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)

            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs >= optimal_threshold).float()  # Use optimal threshold

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # ROC-AUC (handle edge case where only one class is present)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.5  # Default when only one class present

    conf_matrix = confusion_matrix(all_labels, all_preds)

    metrics = {
        'horizon': horizon,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': optimal_threshold,
        'confusion_matrix': conf_matrix,
        'n_samples': len(all_labels),
        'positive_rate': all_labels.mean()
    }

    return metrics


def print_metrics_table(all_metrics):
    """Print a formatted table of metrics across all horizons"""
    print("\n" + "="*120)
    print("PERFORMANCE REPORT: Disruption Prediction Across Horizons")
    print("="*120)
    print(f"{'Horizon':>10} | {'Threshold':>10} | {'Balanced':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'ROC-AUC':>10}")
    print(f"{'':>10} | {'':>10} | {'Accuracy':>10} | {'':>10} | {'':>10} | {'':>10} | {'':>10}")
    print("-"*120)
    
    for m in all_metrics:
        print(f"{m['horizon']:>7} ms | {m['threshold']:>10.2f} | {m['balanced_accuracy']:>10.4f} | {m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['f1']:>10.4f} | {m['roc_auc']:>10.4f}")
    
    print("="*120)
    
    # Additional summary
    print("\nDetailed Confusion Matrices:")
    for m in all_metrics:
        print(f"\n{m['horizon']}ms Horizon (n={m['n_samples']}, positive_rate={m['positive_rate']:.1%}):")
        print(f"  TN: {m['confusion_matrix'][0,0]:>6}  FP: {m['confusion_matrix'][0,1]:>6}")
        print(f"  FN: {m['confusion_matrix'][1,0]:>6}  TP: {m['confusion_matrix'][1,1]:>6}")


def plot_results(all_metrics, save_path=None):
    """Create comprehensive visualization of results across horizons"""
    
    if save_path is None:
        save_path = os.path.join(SCRIPT_DIR, 'disruption_prediction_results.png')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    horizons = [m['horizon'] for m in all_metrics]
    accuracies = [m['accuracy'] for m in all_metrics]
    balanced_accs = [m['balanced_accuracy'] for m in all_metrics]
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    f1_scores = [m['f1'] for m in all_metrics]
    roc_aucs = [m['roc_auc'] for m in all_metrics]
    positive_rates = [m['positive_rate'] for m in all_metrics]

    # Plot 1: Accuracy vs Balanced Accuracy vs Horizon
    axes[0, 0].plot(horizons, accuracies, 'bo-', linewidth=2, markersize=10, label='Accuracy')
    axes[0, 0].plot(horizons, balanced_accs, 'rs-', linewidth=2, markersize=10, label='Balanced Accuracy')
    axes[0, 0].set_xlabel('Prediction Horizon (ms)', fontsize=12)
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Accuracy vs Balanced Accuracy', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # Plot 2: All metrics comparison
    x = np.arange(len(horizons))
    width = 0.2
    axes[0, 1].bar(x - 1.5*width, accuracies, width, label='Accuracy', color='blue', alpha=0.8)
    axes[0, 1].bar(x - 0.5*width, precisions, width, label='Precision', color='green', alpha=0.8)
    axes[0, 1].bar(x + 0.5*width, recalls, width, label='Recall', color='orange', alpha=0.8)
    axes[0, 1].bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='red', alpha=0.8)
    axes[0, 1].set_xlabel('Prediction Horizon (ms)', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Classification Metrics by Horizon', fontsize=14)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(horizons)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1])

    # Plot 3: ROC-AUC vs Horizon
    axes[0, 2].plot(horizons, roc_aucs, 'gs-', linewidth=2, markersize=10)
    axes[0, 2].axhline(y=0.5, color='r', linestyle='--', label='Random Classifier')
    axes[0, 2].set_xlabel('Prediction Horizon (ms)', fontsize=12)
    axes[0, 2].set_ylabel('ROC-AUC', fontsize=12)
    axes[0, 2].set_title('ROC-AUC vs Prediction Horizon', fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0.4, 1])

    # Plot 4: Positive rate (baseline) vs Horizon
    axes[1, 0].bar(horizons, positive_rates, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Prediction Horizon (ms)', fontsize=12)
    axes[1, 0].set_ylabel('Positive Rate', fontsize=12)
    axes[1, 0].set_title('Baseline Positive Rate by Horizon\n(% of samples with disruption)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, (h, p) in enumerate(zip(horizons, positive_rates)):
        axes[1, 0].annotate(f'{p:.1%}', (h, p), textcoords="offset points", 
                           xytext=(0, 5), ha='center', fontsize=9)

    # Plot 5: Precision vs Recall trade-off
    axes[1, 1].scatter(recalls, precisions, c=horizons, cmap='viridis', s=200, edgecolors='black')
    for h, r, p in zip(horizons, recalls, precisions):
        axes[1, 1].annotate(f'{h}ms', (r, p), textcoords="offset points", 
                           xytext=(5, 5), fontsize=9)
    axes[1, 1].set_xlabel('Recall', fontsize=12)
    axes[1, 1].set_ylabel('Precision', fontsize=12)
    axes[1, 1].set_title('Precision-Recall Trade-off by Horizon', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Horizon (ms)')

    # Plot 6: Confusion matrices heatmap (show first and last horizon)
    # Show confusion matrix for 30ms (short) and 200ms (long) horizons
    short_idx = 0  # 30ms
    long_idx = -1  # 200ms
    
    # Create a combined view
    axes[1, 2].set_title('Confusion Matrices: 30ms vs 200ms Horizons', fontsize=14)
    
    # Left half: 30ms
    cm_short = all_metrics[short_idx]['confusion_matrix']
    cm_long = all_metrics[long_idx]['confusion_matrix']
    
    combined_text = f"30ms Horizon:\n"
    combined_text += f"TN={cm_short[0,0]}, FP={cm_short[0,1]}\n"
    combined_text += f"FN={cm_short[1,0]}, TP={cm_short[1,1]}\n\n"
    combined_text += f"200ms Horizon:\n"
    combined_text += f"TN={cm_long[0,0]}, FP={cm_long[0,1]}\n"
    combined_text += f"FN={cm_long[1,0]}, TP={cm_long[1,1]}"
    
    axes[1, 2].text(0.5, 0.5, combined_text, ha='center', va='center', 
                    fontsize=14, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults visualization saved to: {save_path}")


def main():
    """Main training pipeline for Future Disruption Prediction"""
    print("="*70)
    print("Future Disruption Predictor")
    print("="*70)
    print("Task: Predict if disruption (State 3: Mitigated or State 4: ELM)")
    print("      will occur within specified time horizons")
    print("Model: Causal LSTM Binary Classifier (no future data leakage)")
    print(f"Horizons: {HORIZONS} ms")
    print("="*70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Hyperparameters
    window_size = 500  # 500 ms lookback
    batch_size = 512
    n_epochs = 80

    print(f"Window size (lookback): {window_size} ms")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {SCRIPT_DIR}")

    # Load data once
    X, y, shots, times, features, scaler = load_and_prepare_data()
    n_features = len(features)

    # Store results for all horizons
    all_metrics = []
    all_training_history = {}

    # Train separate model for each horizon
    for horizon in HORIZONS:
        print(f"\n{'='*70}")
        print(f"TRAINING FOR {horizon}ms HORIZON")
        print("="*70)

        # Create windows for this horizon
        train_X, train_y, val_X, val_y, test_X, test_y = create_windows_for_horizon(
            X, y, shots, window_size=window_size, horizon=horizon
        )

        print(f"Dataset sizes: Train={len(train_X)}, Val={len(val_X)}, Test={len(test_X)}")

        # Calculate pos_weight for class imbalance
        # pos_weight = num_negatives / num_positives
        num_positives = train_y.sum()
        num_negatives = len(train_y) - num_positives
        if num_positives > 0:
            pos_weight = num_negatives / num_positives
        else:
            pos_weight = 1.0
        print(f"Class balance: {num_negatives:.0f} negatives, {num_positives:.0f} positives")

        # Create data loaders
        train_dataset = PlasmaClassificationDataset(train_X, train_y)
        val_dataset = PlasmaClassificationDataset(val_X, val_y)
        test_dataset = PlasmaClassificationDataset(test_X, test_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)

        # Create fresh model for this horizon
        model = CausalLSTMBinaryClassifier(
            n_features=n_features, 
            lstm_hidden=256, 
            nn_hidden_sizes=[512, 256, 128]
        ).to(device)

        # Train model with class weighting
        model_save_path = os.path.join(SCRIPT_DIR, f'disruption_predictor_{horizon}ms.pth')
        train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, best_val_f1 = train_model_for_horizon(
            model, train_loader, val_loader, device, horizon,
            n_epochs=n_epochs, model_save_path=model_save_path, pos_weight=pos_weight
        )

        # Store training history
        all_training_history[horizon] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'train_f1s': train_f1s,
            'val_f1s': val_f1s
        }

        # Load best model and evaluate (pass val_loader for optimal threshold selection)
        model.load_state_dict(torch.load(model_save_path))
        metrics = evaluate_model_for_horizon(model, test_loader, device, horizon, val_loader=val_loader)
        all_metrics.append(metrics)

        print(f"\n{horizon}ms Test Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    # Print comprehensive results table
    print_metrics_table(all_metrics)

    # Create visualizations
    plot_results(all_metrics)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'horizon_ms': m['horizon'],
        'threshold': m['threshold'],
        'accuracy': m['accuracy'],
        'balanced_accuracy': m['balanced_accuracy'],
        'precision': m['precision'],
        'recall': m['recall'],
        'f1_score': m['f1'],
        'roc_auc': m['roc_auc'],
        'n_samples': m['n_samples'],
        'positive_rate': m['positive_rate']
    } for m in all_metrics])
    
    metrics_csv_path = os.path.join(SCRIPT_DIR, 'disruption_prediction_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nMetrics saved to: {metrics_csv_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nSaved models:")
    for horizon in HORIZONS:
        print(f"  - disruption_predictor_{horizon}ms.pth")
    print(f"\nResults files:")
    print(f"  - disruption_prediction_results.png")
    print(f"  - disruption_prediction_metrics.csv")


if __name__ == "__main__":
    main()
