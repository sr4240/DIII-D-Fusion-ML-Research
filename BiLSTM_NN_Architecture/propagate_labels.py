"""
Label Propagation Script using Trained BiLSTM-NN Model

This script loads a trained model from BiLSTM_NN_Center_Point.py and uses it
to predict/propagate labels for shots in LABEL_PROPAGATED_DATABASE.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Model Definition (must match the training script)
# ============================================================================

class LSTMFirstNN(nn.Module):
    """
    A hybrid model with LSTM processing FIRST (for temporal patterns)
    followed by NN layers (for feature transformation).
    """
    def __init__(self, n_features, n_classes=4, lstm_hidden=128, nn_hidden_sizes=[256, 128]):
        super(LSTMFirstNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        lstm_output_size = lstm_hidden * 2

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

        self.attention_weights = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim + lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
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


class InferenceDataset(Dataset):
    """Dataset for inference (no labels needed)"""
    def __init__(self, windows):
        self.windows = torch.FloatTensor(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].T  # (n_features, sequence_length)


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters
    n_features = checkpoint['n_features']
    n_classes = checkpoint['n_classes']
    lstm_hidden = checkpoint['lstm_hidden']
    nn_hidden_sizes = checkpoint['nn_hidden_sizes']
    
    # Create model
    model = LSTMFirstNN(
        n_features=n_features,
        n_classes=n_classes,
        lstm_hidden=lstm_hidden,
        nn_hidden_sizes=nn_hidden_sizes
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    scaler.n_features_in_ = n_features
    
    print(f"Model loaded successfully!")
    print(f"  Features: {checkpoint['features']}")
    print(f"  Window size: {checkpoint['window_size']}")
    print(f"  Classes: {checkpoint['class_names']}")
    if 'test_accuracy' in checkpoint:
        print(f"  Training test accuracy: {checkpoint['test_accuracy']:.4f}")
    
    return model, scaler, checkpoint


def predict_shot_labels(model, shot_data, scaler, features, window_size, device, batch_size=256):
    """
    Predict labels for a single shot using sliding window approach.
    Only assigns predictions to center points of valid windows.
    Edge points (first window_size//2 and last window_size//2 - 1) remain as -1.
    """
    # Extract and scale features
    X = shot_data[features].values
    
    # Handle NaN values - forward fill then backward fill
    X = pd.DataFrame(X, columns=features).fillna(method='ffill').fillna(method='bfill').values
    
    # Check if still has NaN
    if np.isnan(X).any():
        return np.full(len(shot_data), -1)  # Return -1 for shots with all NaN
    
    # Scale features
    X_scaled = (X - scaler.mean_) / scaler.scale_
    
    n_samples = len(X_scaled)
    center_idx = window_size // 2
    
    # If shot is too short for even one window, assign most common class or -1
    if n_samples < window_size:
        return np.full(n_samples, -1)  # Can't predict, mark as unknown
    
    # Create windows
    windows = []
    window_centers = []  # Track which index each window corresponds to
    
    for i in range(n_samples - window_size + 1):
        window = X_scaled[i:i + window_size]
        if not np.isnan(window).any() and not np.isinf(window).any():
            windows.append(window)
            window_centers.append(i + center_idx)
    
    if len(windows) == 0:
        return np.full(n_samples, -1)
    
    windows = np.array(windows, dtype=np.float32)
    
    # Create dataset and dataloader
    dataset = InferenceDataset(windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Predict
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Create full prediction array for all time points
    # Edge points (first window_size//2 and last window_size//2 - 1) remain as -1
    full_predictions = np.full(n_samples, -1)
    
    # Only assign predictions to center points of valid windows
    for i, center in enumerate(window_centers):
        full_predictions[center] = predictions[i]
    
    return full_predictions


def main():
    print("=" * 60)
    print("Label Propagation using Trained BiLSTM-NN Model")
    print("=" * 60)
    
    # Configuration
    checkpoint_path = '/mnt/homes/sr4240/my_folder/BiLSTM_NN_Architecture/bilstm_nn_complete_model.pth'
    input_csv = '/mnt/homes/sr4240/my_folder/LABEL_PROPAGATED_DATABASE.csv'
    output_csv = input_csv  # Overwrite input file with added state column
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, scaler, checkpoint = load_model(checkpoint_path, device)
    
    features = checkpoint['features']
    window_size = checkpoint['window_size']
    class_names = checkpoint['class_names']
    
    # Load data
    print(f"\nLoading data from: {input_csv}")
    print("This may take a while for large files...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} rows")
    
    # Check if required features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"ERROR: Missing features in data: {missing_features}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"\nRequired features found: {features}")
    
    # Get unique shots
    unique_shots = df['shot'].unique()
    print(f"Total unique shots: {len(unique_shots)}")
    
    # Initialize prediction column
    df['state'] = -1
    
    # Process each shot
    print("\nPredicting labels for each shot...")
    
    shot_stats = {'processed': 0, 'skipped': 0, 'too_short': 0}
    
    for shot_id in tqdm(unique_shots, desc="Processing shots"):
        shot_mask = df['shot'] == shot_id
        shot_data = df.loc[shot_mask].copy()
        
        # Sort by time
        shot_data = shot_data.sort_values('time')
        shot_indices = shot_data.index
        
        # Predict labels
        predictions = predict_shot_labels(
            model, shot_data, scaler, features, window_size, device
        )
        
        # Assign predictions back to dataframe
        if predictions is not None and len(predictions) == len(shot_indices):
            df.loc[shot_indices, 'state'] = predictions
            
            if (predictions == -1).all():
                shot_stats['too_short'] += 1
            else:
                shot_stats['processed'] += 1
        else:
            shot_stats['skipped'] += 1
    
    # Convert 0-indexed predictions to 1-indexed (keep -1 as -1)
    # 0 -> 1 (Suppressed), 1 -> 2 (Dithering), 2 -> 3 (Mitigated), 3 -> 4 (ELMing)
    df.loc[df['state'] >= 0, 'state'] = df.loc[df['state'] >= 0, 'state'] + 1
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Prediction Statistics:")
    print("=" * 60)
    print(f"Shots successfully processed: {shot_stats['processed']}")
    print(f"Shots too short (< {window_size} points): {shot_stats['too_short']}")
    print(f"Shots skipped (errors): {shot_stats['skipped']}")
    
    # Label distribution
    valid_predictions = df['state'][df['state'] >= 1]
    print(f"\nPredicted label distribution:")
    for label, count in sorted(Counter(valid_predictions).items()):
        # Labels are now 1-indexed, so subtract 1 to get class_names index
        class_idx = int(label) - 1
        class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"Class {label}"
        percentage = count / len(valid_predictions) * 100
        print(f"  {label} ({class_name}): {count:,} ({percentage:.1f}%)")
    
    unpredicted = (df['state'] == -1).sum()
    print(f"\nUnpredicted points (marked as -1): {unpredicted:,} ({unpredicted/len(df)*100:.1f}%)")
    
    # Save results - reorder columns to put 'state' in the 3rd position
    cols = df.columns.tolist()
    cols.remove('state')
    cols.insert(2, 'state')  # Insert at index 2 (3rd column)
    df = df[cols]
    
    print(f"\nSaving labeled data to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved successfully!")
    
    print("\n" + "=" * 60)
    print("Label Propagation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

