"""
KNN Hyperparameter Optimization Script

This script performs sequential hyperparameter optimization for the KNN cloud classification model.
It sweeps through different values of n_neighbors and percentile_threshold to find the optimal
combination that maximizes correct classifications within the cloud and minimizes incorrect ones.

The optimization is performed on multiple test shots to ensure robustness.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import os
from datetime import datetime
import json

plt.close('all')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Test shots for optimization (5 different shots)
TEST_SHOTS = [169500, 169501, 169472, 169503, 169504]

# Hyperparameter search ranges
N_NEIGHBORS_RANGE = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]  # Number of neighbors to test
PERCENTILE_THRESHOLD_RANGE = [85.0, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5, 99.7, 99.9]  # Percentile thresholds to test

# KNN feature columns (from original script)
KNN_FEATURE_COLUMNS = [
    "iln3iamp",
    "betan",
    "density",
    "n_eped",
    "li",
    "tritop",
    "fs04_max_smoothed",
]

# Random Forest parameters (fixed from original script)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 35,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Output directory
OUTPUT_DIR = 'KNN_Optimization_Results'

# =============================================================================

def load_plasma_data():
    """Load the plasma data from file"""
    print("Loading plasma data...")

    # Try multiple possible paths
    possible_paths = [
        '../plasma_data.csv',
        'plasma_data.csv',
        '/mnt/homes/sr4240/my_folder/plasma_data.csv'
    ]

    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            print(f"Loaded {len(df)} rows")
            break
        except FileNotFoundError:
            continue

    if df is None:
        raise FileNotFoundError("Could not find plasma_data.csv")

    # Clean the data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['original_index'] = df.index

    return df

def prepare_data_for_shot(df, test_shot, rf_features, target_column='state'):
    """Prepare training and test data, excluding the test shot"""

    # Clean the dataframe
    df_cleaned = df.dropna(subset=rf_features, how='any')
    df_cleaned = df_cleaned[df_cleaned['n'] == 3]

    # Exclude test shot from training
    df_training = df_cleaned[df_cleaned['shot'] != test_shot].copy()
    df_test_shot = df_cleaned[df_cleaned['shot'] == test_shot].copy()

    print(f"  Shot {test_shot}: {len(df_training)} training rows, {len(df_test_shot)} test rows")

    return df_training, df_test_shot

def train_random_forest(df_training, rf_features, target_column='state'):
    """Train Random Forest model on training data"""

    X_train = df_training[rf_features]
    y_train = df_training[target_column]

    # Split for validation
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train_split, y_train_split)

    return clf

def train_knn_cloud(df_training, n_neighbors, percentile_threshold):
    """Train KNN cloud model with specified hyperparameters"""

    # Select KNN features
    knn_features_df = df_training[KNN_FEATURE_COLUMNS].dropna()

    if knn_features_df.empty:
        raise ValueError("No data available for KNN cloud")

    # Fit imputer and scaler
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(knn_features_df.values)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_imputed)

    # Compute distances to neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn.fit(X_train_std)
    distances, _ = nn.kneighbors(X_train_std, return_distance=True)
    train_mean_dists = distances[:, 1:].mean(axis=1)

    # Calculate threshold
    threshold = float(np.percentile(train_mean_dists, percentile_threshold))

    return imputer, scaler, X_train_std, threshold

def classify_with_knn(df_test, imputer, scaler, X_train_std, threshold, n_neighbors):
    """Classify test data points using KNN cloud"""

    # Prepare features
    test_knn_features = df_test[KNN_FEATURE_COLUMNS].copy()
    valid_knn_mask = ~test_knn_features.isnull().any(axis=1)

    # Initialize columns
    df_test = df_test.copy()
    df_test['in_cloud'] = np.nan
    df_test['knn_distance'] = np.nan

    if valid_knn_mask.sum() > 0:
        # Transform valid rows
        X_test = test_knn_features[valid_knn_mask].values
        X_test_imputed = imputer.transform(X_test)
        X_test_std = scaler.transform(X_test_imputed)

        # Compute distances
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(X_train_std)
        distances, _ = nn.kneighbors(X_test_std, return_distance=True)
        mean_dists = distances.mean(axis=1)

        # Classify
        in_cloud = mean_dists <= threshold

        # Store results
        df_test.loc[valid_knn_mask, 'in_cloud'] = in_cloud
        df_test.loc[valid_knn_mask, 'knn_distance'] = mean_dists

    return df_test

def evaluate_performance(df_test, clf, rf_features):
    """Evaluate the performance of the model on test data"""

    # Make RF predictions
    X_test = df_test[rf_features]
    predictions = clf.predict(X_test)
    df_test['predicted_state'] = predictions
    df_test['prediction_correct'] = (df_test['state'] == df_test['predicted_state'])

    # Calculate metrics
    in_cloud = df_test['in_cloud'] == True
    out_cloud = df_test['in_cloud'] == False
    correct = df_test['prediction_correct']

    # Key metrics for optimization
    metrics = {
        'total_points': len(df_test),
        'in_cloud_points': in_cloud.sum(),
        'out_cloud_points': out_cloud.sum(),
        'in_cloud_correct': (correct & in_cloud).sum(),
        'in_cloud_incorrect': ((~correct) & in_cloud).sum(),
        'out_cloud_correct': (correct & out_cloud).sum(),
        'out_cloud_incorrect': ((~correct) & out_cloud).sum(),
    }

    # Calculate percentages
    if metrics['in_cloud_points'] > 0:
        metrics['in_cloud_accuracy'] = metrics['in_cloud_correct'] / metrics['in_cloud_points']
        metrics['in_cloud_error_rate'] = metrics['in_cloud_incorrect'] / metrics['in_cloud_points']
    else:
        metrics['in_cloud_accuracy'] = 0
        metrics['in_cloud_error_rate'] = 1

    if metrics['out_cloud_points'] > 0:
        metrics['out_cloud_accuracy'] = metrics['out_cloud_correct'] / metrics['out_cloud_points']
    else:
        metrics['out_cloud_accuracy'] = 0

    # Overall accuracy
    metrics['overall_accuracy'] = correct.sum() / len(df_test)

    # Optimization score: maximize correct in cloud, minimize incorrect in cloud
    # Higher score is better
    metrics['optimization_score'] = (
        metrics['in_cloud_correct'] * 2.0 -  # Weight correct predictions more
        metrics['in_cloud_incorrect'] * 5 -  # Penalize incorrect predictions heavily
        metrics['out_cloud_correct'] * 0.5 +  # Small penalty for correct out-of-cloud
        metrics['out_cloud_incorrect'] * 1  # bonus for incorrect out-of-cloud
    ) / metrics['total_points']

    return metrics

def sweep_hyperparameters():
    """Perform sequential hyperparameter sweep"""

    print("="*80)
    print("STARTING KNN HYPERPARAMETER OPTIMIZATION")
    print("="*80)

    # Load data once
    df = load_plasma_data()
    rf_features = KNN_FEATURE_COLUMNS

    # Results storage
    all_results = []

    # First sweep: n_neighbors (with default percentile)
    print("\nPHASE 1: Sweeping n_neighbors parameter...")
    print("-"*60)

    default_percentile = 95.0
    best_n_neighbors = None
    best_n_neighbors_score = -float('inf')

    for n_neighbors in N_NEIGHBORS_RANGE:
        print(f"\nTesting n_neighbors = {n_neighbors} (percentile = {default_percentile})")

        shot_scores = []
        shot_metrics = []

        for shot in TEST_SHOTS:
            try:
                # Prepare data
                df_training, df_test_shot = prepare_data_for_shot(df, shot, rf_features)

                # Train Random Forest
                clf = train_random_forest(df_training, rf_features)

                # Train KNN cloud
                imputer, scaler, X_train_std, threshold = train_knn_cloud(
                    df_training, n_neighbors, default_percentile)

                # Classify test shot
                df_test_shot = classify_with_knn(
                    df_test_shot, imputer, scaler, X_train_std, threshold, n_neighbors)

                # Evaluate
                metrics = evaluate_performance(df_test_shot, clf, rf_features)
                metrics['shot'] = shot
                metrics['n_neighbors'] = n_neighbors
                metrics['percentile_threshold'] = default_percentile

                shot_scores.append(metrics['optimization_score'])
                shot_metrics.append(metrics)

                print(f"    Shot {shot}: Score = {metrics['optimization_score']:.4f}, "
                      f"In-cloud acc = {metrics['in_cloud_accuracy']:.3f}, "
                      f"In-cloud err = {metrics['in_cloud_error_rate']:.3f}")

            except Exception as e:
                print(f"    Shot {shot}: Error - {e}")
                continue

        # Calculate average score
        if shot_scores:
            avg_score = np.mean(shot_scores)
            std_score = np.std(shot_scores)
            print(f"  Average score: {avg_score:.4f} ± {std_score:.4f}")

            # Track results
            for m in shot_metrics:
                all_results.append(m)

            # Update best
            if avg_score > best_n_neighbors_score:
                best_n_neighbors_score = avg_score
                best_n_neighbors = n_neighbors
                print(f"  NEW BEST n_neighbors: {n_neighbors}")

    print(f"\nBest n_neighbors: {best_n_neighbors} (score: {best_n_neighbors_score:.4f})")

    # Second sweep: percentile_threshold (with best n_neighbors)
    print("\n" + "="*80)
    print("PHASE 2: Sweeping percentile_threshold parameter...")
    print("-"*60)

    best_percentile = None
    best_percentile_score = -float('inf')

    for percentile in PERCENTILE_THRESHOLD_RANGE:
        print(f"\nTesting percentile = {percentile} (n_neighbors = {best_n_neighbors})")

        shot_scores = []
        shot_metrics = []

        for shot in TEST_SHOTS:
            try:
                # Prepare data
                df_training, df_test_shot = prepare_data_for_shot(df, shot, rf_features)

                # Train Random Forest
                clf = train_random_forest(df_training, rf_features)

                # Train KNN cloud with best n_neighbors
                imputer, scaler, X_train_std, threshold = train_knn_cloud(
                    df_training, best_n_neighbors, percentile)

                # Classify test shot
                df_test_shot = classify_with_knn(
                    df_test_shot, imputer, scaler, X_train_std, threshold, best_n_neighbors)

                # Evaluate
                metrics = evaluate_performance(df_test_shot, clf, rf_features)
                metrics['shot'] = shot
                metrics['n_neighbors'] = best_n_neighbors
                metrics['percentile_threshold'] = percentile

                shot_scores.append(metrics['optimization_score'])
                shot_metrics.append(metrics)

                print(f"    Shot {shot}: Score = {metrics['optimization_score']:.4f}, "
                      f"In-cloud acc = {metrics['in_cloud_accuracy']:.3f}, "
                      f"In-cloud err = {metrics['in_cloud_error_rate']:.3f}")

            except Exception as e:
                print(f"    Shot {shot}: Error - {e}")
                continue

        # Calculate average score
        if shot_scores:
            avg_score = np.mean(shot_scores)
            std_score = np.std(shot_scores)
            print(f"  Average score: {avg_score:.4f} ± {std_score:.4f}")

            # Track results
            for m in shot_metrics:
                all_results.append(m)

            # Update best
            if avg_score > best_percentile_score:
                best_percentile_score = avg_score
                best_percentile = percentile
                print(f"  NEW BEST percentile: {percentile}")

    print(f"\nBest percentile_threshold: {best_percentile} (score: {best_percentile_score:.4f})")

    # Final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nOptimal Hyperparameters:")
    print(f"  n_neighbors: {best_n_neighbors}")
    print(f"  percentile_threshold: {best_percentile}")
    print(f"  Best average score: {best_percentile_score:.4f}")

    return all_results, best_n_neighbors, best_percentile

def create_visualization(results_df):
    """Create visualization of hyperparameter optimization results"""

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('KNN Hyperparameter Optimization Results', fontsize=16, fontweight='bold')

    # Plot 1: n_neighbors vs optimization score
    ax1 = axes[0, 0]
    n_neighbors_grouped = results_df[results_df['percentile_threshold'] == 95.0].groupby('n_neighbors')
    means = n_neighbors_grouped['optimization_score'].mean()
    stds = n_neighbors_grouped['optimization_score'].std()
    ax1.errorbar(means.index, means.values, yerr=stds.values, marker='o', capsize=5)
    ax1.set_xlabel('n_neighbors')
    ax1.set_ylabel('Optimization Score')
    ax1.set_title('n_neighbors Effect (percentile=95)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: percentile vs optimization score
    ax2 = axes[0, 1]
    # Get the best n_neighbors value
    best_n = results_df.groupby('n_neighbors')['optimization_score'].mean().idxmax()
    percentile_grouped = results_df[results_df['n_neighbors'] == best_n].groupby('percentile_threshold')
    means = percentile_grouped['optimization_score'].mean()
    stds = percentile_grouped['optimization_score'].std()
    ax2.errorbar(means.index, means.values, yerr=stds.values, marker='o', capsize=5, color='orange')
    ax2.set_xlabel('Percentile Threshold')
    ax2.set_ylabel('Optimization Score')
    ax2.set_title(f'Percentile Effect (n_neighbors={best_n})')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Heatmap of average scores
    ax3 = axes[0, 2]
    pivot_table = results_df.pivot_table(
        values='optimization_score',
        index='n_neighbors',
        columns='percentile_threshold',
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Score'})
    ax3.set_title('Average Optimization Scores')

    # Plot 4: In-cloud accuracy vs n_neighbors
    ax4 = axes[1, 0]
    n_neighbors_grouped = results_df[results_df['percentile_threshold'] == 95.0].groupby('n_neighbors')
    means = n_neighbors_grouped['in_cloud_accuracy'].mean()
    stds = n_neighbors_grouped['in_cloud_accuracy'].std()
    ax4.errorbar(means.index, means.values, yerr=stds.values, marker='s', capsize=5, color='green')
    ax4.set_xlabel('n_neighbors')
    ax4.set_ylabel('In-Cloud Accuracy')
    ax4.set_title('In-Cloud Accuracy vs n_neighbors')
    ax4.grid(True, alpha=0.3)

    # Plot 5: In-cloud error rate vs percentile
    ax5 = axes[1, 1]
    percentile_grouped = results_df[results_df['n_neighbors'] == best_n].groupby('percentile_threshold')
    means = percentile_grouped['in_cloud_error_rate'].mean()
    stds = percentile_grouped['in_cloud_error_rate'].std()
    ax5.errorbar(means.index, means.values, yerr=stds.values, marker='s', capsize=5, color='red')
    ax5.set_xlabel('Percentile Threshold')
    ax5.set_ylabel('In-Cloud Error Rate')
    ax5.set_title(f'In-Cloud Error Rate vs Percentile (n_neighbors={best_n})')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Shot-by-shot performance with best parameters
    ax6 = axes[1, 2]
    best_params = results_df.groupby(['n_neighbors', 'percentile_threshold'])['optimization_score'].mean().idxmax()
    best_results = results_df[
        (results_df['n_neighbors'] == best_params[0]) &
        (results_df['percentile_threshold'] == best_params[1])
    ]

    x = range(len(best_results))
    ax6.bar(x, best_results['in_cloud_correct'], label='Correct in cloud', color='green', alpha=0.7)
    ax6.bar(x, -best_results['in_cloud_incorrect'], label='Incorrect in cloud', color='red', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels([f"Shot {int(s)}" for s in best_results['shot']], rotation=45)
    ax6.set_ylabel('Number of Points')
    ax6.set_title(f'Best Parameters Performance (n={best_params[0]}, p={best_params[1]})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f'optimization_results_{timestamp}.png')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {filename}")

    return fig

def save_results(results, best_n_neighbors, best_percentile):
    """Save optimization results to files"""

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(OUTPUT_DIR, f'optimization_results_{timestamp}.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to: {csv_filename}")

    # Save best parameters to JSON
    best_params = {
        'n_neighbors': int(best_n_neighbors),
        'percentile_threshold': float(best_percentile),
        'test_shots': TEST_SHOTS,
        'timestamp': timestamp
    }

    json_filename = os.path.join(OUTPUT_DIR, f'best_parameters_{timestamp}.json')
    with open(json_filename, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best parameters saved to: {json_filename}")

    # Also save a "latest" version for easy access
    latest_json = os.path.join(OUTPUT_DIR, 'best_parameters_latest.json')
    with open(latest_json, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Latest parameters saved to: {latest_json}")

    return results_df

def print_summary_statistics(results_df, best_n_neighbors, best_percentile):
    """Print summary statistics of the optimization"""

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Filter for best parameters
    best_results = results_df[
        (results_df['n_neighbors'] == best_n_neighbors) &
        (results_df['percentile_threshold'] == best_percentile)
    ]

    print(f"\nBest Hyperparameters:")
    print(f"  n_neighbors: {best_n_neighbors}")
    print(f"  percentile_threshold: {best_percentile}")

    print(f"\nPerformance with Best Parameters:")
    print(f"  Average optimization score: {best_results['optimization_score'].mean():.4f}")
    print(f"  Average in-cloud accuracy: {best_results['in_cloud_accuracy'].mean():.3f}")
    print(f"  Average in-cloud error rate: {best_results['in_cloud_error_rate'].mean():.3f}")
    print(f"  Average overall accuracy: {best_results['overall_accuracy'].mean():.3f}")

    print(f"\nDetailed Breakdown:")
    for _, row in best_results.iterrows():
        print(f"  Shot {int(row['shot'])}:")
        print(f"    In-cloud: {int(row['in_cloud_correct'])} correct, {int(row['in_cloud_incorrect'])} incorrect")
        print(f"    Out-cloud: {int(row['out_cloud_correct'])} correct, {int(row['out_cloud_incorrect'])} incorrect")
        print(f"    Score: {row['optimization_score']:.4f}")

    # Compare to baseline (default parameters)
    baseline = results_df[
        (results_df['n_neighbors'] == 20) &
        (results_df['percentile_threshold'] == 95.0)
    ]

    if not baseline.empty:
        print(f"\nImprovement over baseline (n=20, p=95):")
        baseline_score = baseline['optimization_score'].mean()
        best_score = best_results['optimization_score'].mean()
        improvement = ((best_score - baseline_score) / abs(baseline_score)) * 100
        print(f"  Baseline score: {baseline_score:.4f}")
        print(f"  Optimized score: {best_score:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")

def main():
    """Main function to run the optimization"""

    try:
        # Run hyperparameter sweep
        results, best_n_neighbors, best_percentile = sweep_hyperparameters()

        # Save results
        results_df = save_results(results, best_n_neighbors, best_percentile)

        # Create visualization
        create_visualization(results_df)

        # Print summary
        print_summary_statistics(results_df, best_n_neighbors, best_percentile)

        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE!")
        print(f"Results saved in: {OUTPUT_DIR}/")
        print("="*80)

        return results_df, best_n_neighbors, best_percentile

    except Exception as e:
        print(f"\nError during optimization: {e}")
        raise

if __name__ == "__main__":
    results_df, best_n, best_p = main()