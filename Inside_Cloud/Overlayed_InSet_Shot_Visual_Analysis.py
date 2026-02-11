"""
Overlayed In-Set Shot Visualization Script with kNN Cloud Classification

This script visualizes Random Forest predictions for a specified shot while also overlaying
the kNN cloud classification to show which points are inside or outside the training data cloud.
The shot IS INCLUDED in the training data - this shows how the model performs on data it was trained on.

HOW TO USE:
1. Change the SHOT_NUMBER variable below to the shot you want to analyze
2. Run the script: python Overlayed_InSet_Shot_Visual_Analysis.py
3. The script will generate visualization files showing both RF predictions and kNN cloud classification

The script combines Random Forest prediction accuracy with kNN cloud distance metrics to provide
a comprehensive view of model performance and data distribution for in-sample predictions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import os

plt.close('all')

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS AS NEEDED
# =============================================================================

# Shot number to analyze (change this to any shot number you want to visualize)
# Examples: 169501, 169472, 169503, etc.
SHOT_NUMBER = 169472

# kNN parameters
N_NEIGHBORS = 30  # Number of nearest neighbors for kNN cloud
PERCENTILE_THRESHOLD = 95.0  # Percentile for determining cloud boundary

# Output file prefix (will be automatically updated based on shot number)
OUTPUT_PREFIX = f"overlayed_inset_shot_{SHOT_NUMBER}"

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

def load_and_train_model_including_shot(include_shot_number = SHOT_NUMBER):
    """Load data and train the Random Forest model while INCLUDING the specified shot in training"""
    print("Loading plasma data...")

    # Try multiple possible paths for the plasma data file
    possible_paths = [
        '../plasma_data.csv',  # Relative path from Random_Forest directory
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

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Add an index column to keep track of original row numbers
    df['original_index'] = df.index

    # Select features and the target variable for Random Forest
    rf_selected_features = ['iln3iamp', 'betan', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed']
    target_column = 'state'

    # Clean the dataframe
    df_cleaned = df.dropna(subset=rf_selected_features, how='any')
    df_cleaned = df_cleaned[df_cleaned['n'] == 3]

    # CRITICAL: Include all data (including target shot) in training
    print(f"INCLUDING shot {include_shot_number} in training data...")
    df_training = df_cleaned.copy()
    df_target_shot = df_cleaned[df_cleaned['shot'] == include_shot_number].copy()

    print(f"Training data: {len(df_training)} rows (INCLUDING shot {include_shot_number})")
    print(f"Target shot data: {len(df_target_shot)} rows (shot {include_shot_number})")

    # Prepare input (X) and target (y) for training (including target shot)
    X_train = df_training[rf_selected_features]
    y_train = df_training[target_column]

    # Split training data into train/test for model validation
    X_train_split, X_test_split, y_train_split, y_test_split, idx_train_split, idx_test_split = train_test_split(
        X_train, y_train, df_training['original_index'], test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    print("Training Random Forest model on data including target shot...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=35, min_samples_split=2,
                                min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1)
    clf.fit(X_train_split, y_train_split)

    # Calculate accuracy on the held-out training test set
    y_pred_split = clf.predict(X_test_split)
    accuracy = accuracy_score(y_test_split, y_pred_split)
    print(f"Model accuracy on training test set: {accuracy:.3f}")

    return clf, rf_selected_features, df_cleaned, df_target_shot, df_training

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

def make_predictions_on_shot(clf, shot_data, selected_features):
    """Make predictions on the shot data"""
    print("Making predictions on in-set shot data...")

    # Prepare features for prediction
    X_shot = shot_data[selected_features]

    # Make predictions
    predictions = clf.predict(X_shot)
    prediction_proba = clf.predict_proba(X_shot)

    # Add predictions to shot data
    shot_data = shot_data.copy()
    shot_data['predicted_state'] = predictions
    shot_data['prediction_confidence'] = np.max(prediction_proba, axis=1)

    # Determine if predictions are correct
    shot_data['prediction_correct'] = (shot_data['state'] == shot_data['predicted_state'])

    # Calculate accuracy for this shot
    shot_accuracy = shot_data['prediction_correct'].mean()
    print(f"Prediction accuracy for in-set shot {shot_data['shot'].iloc[0]}: {shot_accuracy:.3f}")

    return shot_data

def create_overlayed_visualization(shot_data):
    """Create comprehensive visualization with kNN cloud overlay"""
    print("Creating overlayed visualization...")

    # Create figure with subplots (3 plots, removed kNN distance graph)
    fig, axes = plt.subplots(3, 1, figsize=(24, 18))
    fig.suptitle(f'In-Set Shot {SHOT_NUMBER} Analysis with kNN Cloud Classification\n(Shot INCLUDED in training)',
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

    # Color background by predicted state
    for state in [1, 2, 3, 4]:
        state_mask = predicted_states == state
        if state_mask.any():
            ax2.fill_between(time, fs04.min(), fs04.max(),
                           where=state_mask, alpha=0.3,
                           color=state_colors[state],
                           label=f'Predicted: {state_names[state]}')

    ax2.set_ylabel('fs04', fontsize=16)
    ax2.set_title('fs04 Time Series with Predicted States', fontsize=18, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    # Plot 3: fs04 with correct/incorrect predictions overlay
    ax3 = axes[2]

    # Plot fs04 time series
    ax3.plot(time, fs04, 'k-', linewidth=2, alpha=0.8, label='fs04')

    # Create mask for correct/incorrect predictions
    correct_mask = shot_data['prediction_correct']
    incorrect_mask = ~correct_mask

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

    # Add markers for prediction confidence (no colorbar/legend)
    confidence = shot_data['prediction_confidence']
    scatter = ax3.scatter(time, fs04, c=confidence, cmap='viridis',
                         s=40, alpha=0.9, edgecolors='none', linewidth=0)

    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04', fontsize=16)
    ax3.set_title('Prediction Accuracy with Confidence Overlay', fontsize=18, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, hspace=0.3, bottom=0.06, top=0.96)

    return fig

def create_detailed_analysis_plot(shot_data):
    """Create a detailed analysis plot focusing on the relationship between predictions and cloud classification"""
    print("Creating detailed analysis plot...")

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle(f'Shot {SHOT_NUMBER}: RF Predictions vs kNN Cloud Analysis\n(Shot INCLUDED in training)',
                 fontsize=20, fontweight='bold')

    time = shot_data['time']
    fs04 = shot_data['fs04']
    actual_states = shot_data['state']
    predicted_states = shot_data['predicted_state']
    correct_predictions = shot_data['prediction_correct']
    in_cloud = shot_data['in_cloud'] == True
    out_cloud = shot_data['in_cloud'] == False

    # Plot 1: State transitions with cloud indicators
    ax1 = axes[0]

    # Plot actual and predicted states
    ax1.step(time, actual_states, 'b-', linewidth=2, label='Actual State', where='post')
    ax1.step(time, predicted_states, 'r--', linewidth=2, label='Predicted State', where='post')

    # Mark points outside cloud with vertical lines
    out_cloud_times = time[out_cloud]
    for t in out_cloud_times[::5]:  # Plot every 5th point to avoid clutter
        ax1.axvline(x=t, color='orange', alpha=0.2, linewidth=0.5)

    # Highlight incorrect predictions
    incorrect_times = time[~correct_predictions]
    incorrect_actual = actual_states[~correct_predictions]
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

    # Calculate accuracy metrics
    categories = ['In Cloud\nCorrect', 'In Cloud\nIncorrect', 'Out Cloud\nCorrect', 'Out Cloud\nIncorrect']
    counts = [
        (correct_predictions & in_cloud).sum(),
        ((~correct_predictions) & in_cloud).sum(),
        (correct_predictions & out_cloud).sum(),
        ((~correct_predictions) & out_cloud).sum()
    ]

    colors_bar = ['green', 'red', 'lightgreen', 'lightcoral']
    bars = ax2.bar(categories, counts, color=colors_bar, edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    total_in_cloud = in_cloud.sum()
    total_out_cloud = out_cloud.sum()

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
    ax2.set_title('Prediction Accuracy by Cloud Membership', fontsize=18, fontweight='bold')
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
        elif correct_predictions.iloc[i] and in_cloud.iloc[i]:
            combined_colors.append('green')
            combined_labels.append('Correct + In Cloud')
        elif correct_predictions.iloc[i] and out_cloud.iloc[i]:
            combined_colors.append('lightgreen')
            combined_labels.append('Correct + Out Cloud')
        elif not correct_predictions.iloc[i] and in_cloud.iloc[i]:
            combined_colors.append('red')
            combined_labels.append('Incorrect + In Cloud')
        else:  # incorrect and out of cloud
            combined_colors.append('orange')
            combined_labels.append('Incorrect + Out Cloud')

    # Scatter plot with combined classification
    scatter = ax3.scatter(time, fs04, c=combined_colors, s=30, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add legend with unique labels
    unique_combinations = []
    unique_colors = []
    for color, label in zip(['green', 'lightgreen', 'red', 'orange', 'gray'],
                           ['Correct + In Cloud', 'Correct + Out Cloud',
                            'Incorrect + In Cloud', 'Incorrect + Out Cloud', 'No kNN data']):
        if color in combined_colors:
            unique_combinations.append(plt.scatter([], [], c=color, s=50, edgecolors='black', linewidth=0.5))
            unique_colors.append(label)

    ax3.legend(unique_combinations, unique_colors, fontsize=12, loc='upper right')

    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04', fontsize=16)
    ax3.set_title('Combined RF Accuracy and kNN Cloud Classification', fontsize=18, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Ensure proper layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.94, hspace=0.35)

    return fig

def print_detailed_statistics(shot_data):
    """Print detailed statistics about predictions and cloud classification"""
    print("\n" + "="*70)
    print(f"OVERLAYED ANALYSIS FOR SHOT {shot_data['shot'].iloc[0]}")
    print(f"(Shot {shot_data['shot'].iloc[0]} was INCLUDED in training data)")
    print("="*70)

    # Overall RF statistics
    total_points = len(shot_data)
    correct_predictions = shot_data['prediction_correct'].sum()
    accuracy = correct_predictions / total_points

    print(f"\nRandom Forest Prediction Statistics:")
    print(f"  Total time points: {total_points}")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Incorrect predictions: {total_points - correct_predictions}")
    print(f"  Overall accuracy: {accuracy:.3f}")

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

    if in_cloud.sum() > 0:
        in_cloud_correct = (shot_data['prediction_correct'] & in_cloud).sum()
        in_cloud_accuracy = in_cloud_correct / in_cloud.sum()
        print(f"  In-cloud accuracy: {in_cloud_correct}/{in_cloud.sum()} ({in_cloud_accuracy:.3f})")

    if out_cloud.sum() > 0:
        out_cloud_correct = (shot_data['prediction_correct'] & out_cloud).sum()
        out_cloud_accuracy = out_cloud_correct / out_cloud.sum()
        print(f"  Out-of-cloud accuracy: {out_cloud_correct}/{out_cloud.sum()} ({out_cloud_accuracy:.3f})")

    # Breakdown by category
    print(f"\nDetailed Breakdown:")
    print(f"  Correct + In Cloud: {(shot_data['prediction_correct'] & in_cloud).sum()}")
    print(f"  Correct + Out Cloud: {(shot_data['prediction_correct'] & out_cloud).sum()}")
    print(f"  Incorrect + In Cloud: {((~shot_data['prediction_correct']) & in_cloud).sum()}")
    print(f"  Incorrect + Out Cloud: {((~shot_data['prediction_correct']) & out_cloud).sum()}")

    # Statistics by actual state
    print(f"\nAccuracy by Actual State:")
    for state in [1, 2, 3, 4]:
        state_mask = shot_data['state'] == state
        if state_mask.any():
            state_data = shot_data[state_mask]
            state_correct = state_data['prediction_correct'].sum()
            state_total = len(state_data)
            state_accuracy = state_correct / state_total

            state_in_cloud = (state_data['in_cloud'] == True).sum()
            state_out_cloud = (state_data['in_cloud'] == False).sum()

            state_name = {1: 'Suppressed', 2: 'Dithering', 3: 'Mitigated', 4: 'ELMing'}[state]
            print(f"  {state_name} (State {state}):")
            print(f"    Accuracy: {state_correct}/{state_total} ({state_accuracy:.3f})")
            print(f"    In cloud: {state_in_cloud}, Out of cloud: {state_out_cloud}")

def main():
    """Main function to run the complete analysis"""
    print(f"Starting Overlayed In-Set Shot {SHOT_NUMBER} Visualization Analysis...")
    print("="*70)
    print(f"IMPORTANT: Shot {SHOT_NUMBER} will be INCLUDED in training data")
    print("="*70)

    # Load data and train Random Forest model (including target shot)
    clf, rf_selected_features, df_cleaned, df_target_shot, df_training = load_and_train_model_including_shot(SHOT_NUMBER)

    # Train kNN cloud model
    imputer, scaler, X_train_std, threshold = train_knn_cloud(df_training)

    # Check if shot exists
    if not check_shot_exists(df_cleaned, SHOT_NUMBER):
        print("\nTo change the shot number, edit the SHOT_NUMBER variable at the top of this script.")
        return None

    # Extract shot data
    shot_data = extract_shot_data(df_cleaned, shot_number=SHOT_NUMBER)

    # Make Random Forest predictions
    shot_data = make_predictions_on_shot(clf, shot_data, rf_selected_features)

    # Classify with kNN cloud
    shot_data = classify_shot_with_knn_cloud(shot_data, imputer, scaler, X_train_std, threshold)

    # Create visualizations
    fig1 = create_overlayed_visualization(shot_data)
    fig2 = create_detailed_analysis_plot(shot_data)

    # Print detailed statistics
    print_detailed_statistics(shot_data)

    # Create output directory if it doesn't exist
    output_dir = 'Overlayed_Analysis'
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
