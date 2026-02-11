"""
Random Forest Baseline for Disruption Prediction
=================================================
Tests if features contain predictive signal for the disruption prediction task.
Uses aggregated window features instead of raw sequences.

If RF gets ROC-AUC > 0.6: Features have signal, LSTM architecture needs work
If RF gets ROC-AUC ~0.5: Features don't predict transitions, need different approach
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score,
                             balanced_accuracy_score)
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prediction horizons
HORIZONS = [30, 50, 80, 100, 120, 150, 200]

# States
SUPPRESSED_STATE = 1
DISRUPTION_STATES = [2, 3, 4]


def aggregate_window_features(window):
    """
    Convert a (window_size, n_features) window into an aggregated feature vector.
    This captures summary statistics that RF can work with.
    """
    features = []
    n_features = window.shape[1]
    
    for col in range(n_features):
        signal = window[:, col]
        
        # Basic statistics
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.median(signal))
        
        # Current value (end of window)
        features.append(signal[-1])
        
        # Trend (difference between end and start)
        features.append(signal[-1] - signal[0])
        
        # Rate of change (mean of differences)
        features.append(np.mean(np.diff(signal)))
        
        # Recent values (last 10, 30, 50 timesteps)
        features.append(np.mean(signal[-10:]))
        features.append(np.mean(signal[-30:]))
        features.append(np.mean(signal[-50:]))
        
        # Variance in recent window
        features.append(np.std(signal[-30:]))
        
        # Percentiles
        features.append(np.percentile(signal, 25))
        features.append(np.percentile(signal, 75))
        
        # Skewness approximation
        mean_val = np.mean(signal)
        std_val = np.std(signal) + 1e-8
        features.append(np.mean(((signal - mean_val) / std_val) ** 3))
        
    return np.array(features)


def load_and_prepare_data():
    """Load and preprocess the plasma data"""
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    # Select features
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
    """Check if transition from Suppressed to disruption occurs within horizon"""
    if states[current_idx] != SUPPRESSED_STATE:
        return -1
    
    for i in range(1, horizon + 1):
        future_idx = current_idx + i
        if future_idx >= len(states):
            break
        if states[future_idx] in DISRUPTION_STATES:
            return 1
    
    return 0


def create_windows_for_horizon(X, y, shots, window_size=500, horizon=100):
    """Create windows and aggregate features for Random Forest"""
    print(f"\nCreating windows for {horizon}ms horizon...")

    windows = []
    labels = []
    window_shots = []

    unique_shots = np.unique(shots)
    
    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        effective_window = min(window_size, len(shot_indices))
        
        if effective_window < 50:
            continue

        shot_states = y[shot_indices]

        for i in range(effective_window - 1, len(shot_indices) - horizon):
            start_idx = shot_indices[i - effective_window + 1]
            end_idx = shot_indices[i] + 1

            window = X[start_idx:end_idx]
            current_state = y[shot_indices[i]]

            if current_state != SUPPRESSED_STATE:
                continue

            current_pos_in_shot = i
            label = check_transition_in_horizon(shot_states, current_pos_in_shot, horizon)
            
            if label == -1:
                continue

            # Pad window if needed
            if len(window) < window_size:
                padding = np.zeros((window_size - len(window), window.shape[1]))
                window = np.vstack([padding, window])

            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(label)
                window_shots.append(shot_id)

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    window_shots = np.array(window_shots)

    print(f"  Created {len(windows)} windows")
    print(f"  Label distribution: {Counter(labels)}")
    positive_rate = labels.mean() * 100
    print(f"  Transition rate: {positive_rate:.1f}%")

    # Temporal split
    unique_shots_sorted = np.sort(np.unique(window_shots))
    n_shots = len(unique_shots_sorted)

    train_shot_end = int(0.7 * n_shots)
    val_shot_end = int(0.85 * n_shots)

    train_shots = set(unique_shots_sorted[:train_shot_end])
    val_shots = set(unique_shots_sorted[train_shot_end:val_shot_end])
    test_shots = set(unique_shots_sorted[val_shot_end:])

    train_mask = np.array([s in train_shots for s in window_shots])
    val_mask = np.array([s in val_shots for s in window_shots])
    test_mask = np.array([s in test_shots for s in window_shots])

    return (windows[train_mask], labels[train_mask],
            windows[val_mask], labels[val_mask],
            windows[test_mask], labels[test_mask])


def train_and_evaluate_rf(train_X, train_y, val_X, val_y, test_X, test_y, horizon):
    """Train Random Forest and evaluate"""
    print(f"\n  Aggregating window features for RF...")
    
    # Aggregate windows to feature vectors
    train_features = np.array([aggregate_window_features(w) for w in train_X])
    val_features = np.array([aggregate_window_features(w) for w in val_X])
    test_features = np.array([aggregate_window_features(w) for w in test_X])
    
    print(f"  Feature vector size: {train_features.shape[1]}")
    print(f"  Training RF with {len(train_features)} samples...")
    
    # Train Random Forest with class balancing
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    rf.fit(train_features, train_y)
    
    # Get predictions and probabilities
    train_probs = rf.predict_proba(train_features)[:, 1]
    val_probs = rf.predict_proba(val_features)[:, 1]
    test_probs = rf.predict_proba(test_features)[:, 1]
    
    # Find optimal threshold on validation set
    best_f1 = 0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (val_probs >= thresh).astype(float)
        f1 = f1_score(val_y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"  Optimal threshold: {best_threshold:.2f} (Val F1: {best_f1:.4f})")
    
    # Evaluate on test set with optimal threshold
    test_preds = (test_probs >= best_threshold).astype(float)
    
    accuracy = accuracy_score(test_y, test_preds)
    balanced_acc = balanced_accuracy_score(test_y, test_preds)
    precision = precision_score(test_y, test_preds, zero_division=0)
    recall = recall_score(test_y, test_preds, zero_division=0)
    f1 = f1_score(test_y, test_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(test_y, test_probs)
    except ValueError:
        roc_auc = 0.5
    
    conf_matrix = confusion_matrix(test_y, test_preds)
    
    # Feature importance (top 10)
    feature_importance = rf.feature_importances_
    
    return {
        'horizon': horizon,
        'threshold': best_threshold,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'n_test': len(test_y),
        'positive_rate': test_y.mean(),
        'feature_importance': feature_importance
    }


def print_results(all_metrics):
    """Print results table"""
    print("\n" + "="*100)
    print("RANDOM FOREST BASELINE RESULTS")
    print("="*100)
    print(f"{'Horizon':>10} | {'Threshold':>10} | {'Balanced':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'ROC-AUC':>10}")
    print(f"{'':>10} | {'':>10} | {'Accuracy':>10} | {'':>10} | {'':>10} | {'':>10} | {'':>10}")
    print("-"*100)
    
    for m in all_metrics:
        print(f"{m['horizon']:>7} ms | {m['threshold']:>10.2f} | {m['balanced_accuracy']:>10.4f} | {m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['f1']:>10.4f} | {m['roc_auc']:>10.4f}")
    
    print("="*100)
    
    # Interpretation
    avg_roc_auc = np.mean([m['roc_auc'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if avg_roc_auc > 0.65:
        print("ROC-AUC > 0.65: Features HAVE predictive signal!")
        print("-> LSTM architecture/training needs improvement")
    elif avg_roc_auc > 0.55:
        print("ROC-AUC 0.55-0.65: Features have WEAK signal")
        print("-> May need more/better features, or task reformulation")
    else:
        print("ROC-AUC < 0.55: Features have NO useful signal")
        print("-> This task may not be learnable from these features")
        print("-> Consider: different features, longer horizons, or different task")
    
    print(f"\nAverage ROC-AUC: {avg_roc_auc:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    
    # Confusion matrices
    print("\nConfusion Matrices:")
    for m in all_metrics:
        print(f"\n{m['horizon']}ms (n={m['n_test']}, pos_rate={m['positive_rate']:.1%}):")
        print(f"  TN: {m['confusion_matrix'][0,0]:>6}  FP: {m['confusion_matrix'][0,1]:>6}")
        print(f"  FN: {m['confusion_matrix'][1,0]:>6}  TP: {m['confusion_matrix'][1,1]:>6}")


def main():
    """Main function"""
    print("="*60)
    print("Random Forest Baseline for Disruption Prediction")
    print("="*60)
    print("Purpose: Test if features contain predictive signal")
    print("="*60)

    # Load data
    X, y, shots, times, features, scaler = load_and_prepare_data()

    all_metrics = []

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon}ms")
        print("="*60)

        # Create windows
        train_X, train_y, val_X, val_y, test_X, test_y = create_windows_for_horizon(
            X, y, shots, window_size=500, horizon=horizon
        )

        print(f"  Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")

        # Train and evaluate RF
        metrics = train_and_evaluate_rf(
            train_X, train_y, val_X, val_y, test_X, test_y, horizon
        )
        all_metrics.append(metrics)

        print(f"\n  {horizon}ms Results:")
        print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"    F1-Score: {metrics['f1']:.4f}")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")

    # Print final results
    print_results(all_metrics)

    # Save results
    results_df = pd.DataFrame([{
        'horizon_ms': m['horizon'],
        'threshold': m['threshold'],
        'balanced_accuracy': m['balanced_accuracy'],
        'precision': m['precision'],
        'recall': m['recall'],
        'f1_score': m['f1'],
        'roc_auc': m['roc_auc'],
        'n_test': m['n_test'],
        'positive_rate': m['positive_rate']
    } for m in all_metrics])
    
    csv_path = os.path.join(SCRIPT_DIR, 'rf_baseline_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()