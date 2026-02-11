"""
Hypersphere Hyperparameter Optimization Script

This script performs hyperparameter optimization for the hypersphere (fractional minimum enclosing ball)
cloud classification model. It optimizes the radius_pct parameter to find the optimal percentile threshold
that maximizes correct classifications within the cloud and minimizes incorrect ones.

The optimization uses grid search across multiple test shots to ensure robustness.
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
import os
from datetime import datetime
import json
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Import hypersphere functions from plasma_hypersphere
import sys
sys.path.append('/mnt/homes/sr4240/my_folder/Inside_Cloud')
from plasma_hypersphere import (
    select_features,
    fit_imputer_and_standardizer,
    compute_fractional_meb,
    geometric_median,
    FEATURE_COLUMNS
)

plt.close('all')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Test shots for optimization (5 different shots)
TEST_SHOTS = [169500, 169501, 169472, 169503, 169504]

# Hyperparameter search range for radius_pct (percentile threshold)
# More granular search space for better optimization
RADIUS_PCT_RANGE = [
    80.0, 82.5, 85.0, 87.5, 90.0,
    92.5, 93.0, 93.5, 94.0, 94.5,
    95.0, 95.5, 96.0, 96.5, 97.0,
    97.5, 98.0, 98.5, 99.0, 99.5, 99.7, 99.9
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
OUTPUT_DIR = 'Hypersphere_Optimization_Results'

# =============================================================================

def load_plasma_data() -> pd.DataFrame:
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

def prepare_data_for_shot(df: pd.DataFrame, test_shot: int,
                         rf_features: List[str], target_column: str = 'state') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training and test data, excluding the test shot"""

    # Clean the dataframe
    df_cleaned = df.dropna(subset=rf_features, how='any')
    df_cleaned = df_cleaned[df_cleaned['n'] == 3]

    # Exclude test shot from training
    df_training = df_cleaned[df_cleaned['shot'] != test_shot].copy()
    df_test_shot = df_cleaned[df_cleaned['shot'] == test_shot].copy()

    print(f"  Shot {test_shot}: {len(df_training)} training rows, {len(df_test_shot)} test rows")

    return df_training, df_test_shot

def train_random_forest(df_training: pd.DataFrame, rf_features: List[str],
                       target_column: str = 'state') -> RandomForestClassifier:
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

def train_hypersphere_cloud(df_training: pd.DataFrame, radius_pct: float) -> Tuple:
    """Train hypersphere cloud model with specified radius percentile"""

    # Select features using the function from plasma_hypersphere
    features_df, feature_cols = select_features(df_training)

    if features_df.empty:
        raise ValueError("No data available for hypersphere cloud")

    # Fit imputer and scaler
    imputer, scaler, X_train_std = fit_imputer_and_standardizer(features_df)

    # Compute fractional minimum enclosing ball
    center, radius = compute_fractional_meb(X_train_std, radius_pct)

    return imputer, scaler, center, radius, feature_cols

def classify_with_hypersphere(df_test: pd.DataFrame, imputer: SimpleImputer,
                             scaler: StandardScaler, center: np.ndarray,
                             radius: float, feature_cols: List[str]) -> pd.DataFrame:
    """Classify test data points using hypersphere cloud"""

    # Prepare features
    test_features = df_test[feature_cols].copy()
    valid_mask = ~test_features.isnull().any(axis=1)

    # Initialize columns
    df_test = df_test.copy()
    df_test['in_cloud'] = np.nan
    df_test['distance_to_center'] = np.nan

    if valid_mask.sum() > 0:
        # Transform valid rows
        X_test = test_features[valid_mask].values
        X_test_imputed = imputer.transform(X_test)
        X_test_std = scaler.transform(X_test_imputed)

        # Compute distances to center
        distances = np.linalg.norm(X_test_std - center[None, :], axis=1)

        # Classify based on radius
        in_cloud = distances <= radius

        # Store results
        df_test.loc[valid_mask, 'in_cloud'] = in_cloud
        df_test.loc[valid_mask, 'distance_to_center'] = distances

    return df_test

def evaluate_performance(df_test: pd.DataFrame, clf: RandomForestClassifier,
                        rf_features: List[str]) -> Dict:
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

    # Calculate cloud coverage (percentage of points in cloud)
    metrics['cloud_coverage'] = metrics['in_cloud_points'] / metrics['total_points']

    # Optimization score: balance between coverage and accuracy
    # We want high accuracy in cloud, reasonable coverage, and low error rate
    metrics['optimization_score'] = (
        metrics['in_cloud_correct'] * 2.0 -  # Weight correct predictions more
        metrics['in_cloud_incorrect'] * 8 -  # Penalize incorrect predictions heavily
        metrics['out_cloud_correct'] * 0.5 +  # Small penalty for correct out-of-cloud
        metrics['out_cloud_incorrect'] * 1 # bonus for incorrect out-of-cloud
    ) / metrics['total_points']

    # Add mean distance for points in cloud
    if in_cloud.sum() > 0:
        metrics['mean_distance_in_cloud'] = df_test.loc[in_cloud, 'distance_to_center'].mean()
    else:
        metrics['mean_distance_in_cloud'] = np.nan

    return metrics

def grid_search_optimization() -> Tuple[List[Dict], float]:
    """Perform grid search hyperparameter optimization"""

    print("="*80)
    print("STARTING HYPERSPHERE HYPERPARAMETER OPTIMIZATION")
    print("="*80)

    # Load data once
    df = load_plasma_data()
    rf_features = FEATURE_COLUMNS

    # Results storage
    all_results = []

    print("\nPerforming Grid Search on radius_pct parameter...")
    print("-"*60)

    best_radius_pct = None
    best_score = -float('inf')

    # Grid search over all radius_pct values
    for radius_pct in RADIUS_PCT_RANGE:
        print(f"\nTesting radius_pct = {radius_pct:.1f}%")

        shot_scores = []
        shot_metrics = []

        for shot in TEST_SHOTS:
            try:
                # Prepare data
                df_training, df_test_shot = prepare_data_for_shot(df, shot, rf_features)

                # Train Random Forest
                clf = train_random_forest(df_training, rf_features)

                # Train hypersphere cloud
                imputer, scaler, center, radius, feature_cols = train_hypersphere_cloud(
                    df_training, radius_pct)

                # Classify test shot
                df_test_shot = classify_with_hypersphere(
                    df_test_shot, imputer, scaler, center, radius, feature_cols)

                # Evaluate
                metrics = evaluate_performance(df_test_shot, clf, rf_features)
                metrics['shot'] = shot
                metrics['radius_pct'] = radius_pct
                metrics['radius_value'] = radius  # Store actual radius value

                shot_scores.append(metrics['optimization_score'])
                shot_metrics.append(metrics)

                print(f"    Shot {shot}: Score = {metrics['optimization_score']:.4f}, "
                      f"In-cloud acc = {metrics['in_cloud_accuracy']:.3f}, "
                      f"Coverage = {metrics['cloud_coverage']:.3f}")

            except Exception as e:
                print(f"    Shot {shot}: Error - {e}")
                continue

        # Calculate average score
        if shot_scores:
            avg_score = np.mean(shot_scores)
            std_score = np.std(shot_scores)
            avg_coverage = np.mean([m['cloud_coverage'] for m in shot_metrics])
            avg_in_cloud_acc = np.mean([m['in_cloud_accuracy'] for m in shot_metrics])

            print(f"  Average: Score = {avg_score:.4f} ± {std_score:.4f}, "
                  f"Coverage = {avg_coverage:.3f}, In-cloud acc = {avg_in_cloud_acc:.3f}")

            # Track results
            for m in shot_metrics:
                all_results.append(m)

            # Update best
            if avg_score > best_score:
                best_score = avg_score
                best_radius_pct = radius_pct
                print(f"  >>> NEW BEST radius_pct: {radius_pct:.1f}%")

    # Final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nOptimal Hyperparameter:")
    print(f"  radius_pct: {best_radius_pct:.1f}%")
    print(f"  Best average score: {best_score:.4f}")

    return all_results, best_radius_pct

def create_comprehensive_visualization(results_df: pd.DataFrame) -> plt.Figure:
    """Create comprehensive visualization of optimization results"""

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Hypersphere Hyperparameter Optimization Results', fontsize=18, fontweight='bold')

    # 1. Optimization score vs radius_pct
    ax1 = fig.add_subplot(gs[0, 0])
    grouped = results_df.groupby('radius_pct')
    means = grouped['optimization_score'].mean()
    stds = grouped['optimization_score'].std()
    ax1.errorbar(means.index, means.values, yerr=stds.values, marker='o', capsize=5, linewidth=2)
    ax1.set_xlabel('Radius Percentile (%)')
    ax1.set_ylabel('Optimization Score')
    ax1.set_title('Optimization Score vs Radius Percentile')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 2. In-cloud accuracy vs radius_pct
    ax2 = fig.add_subplot(gs[0, 1])
    means = grouped['in_cloud_accuracy'].mean()
    stds = grouped['in_cloud_accuracy'].std()
    ax2.errorbar(means.index, means.values, yerr=stds.values, marker='s', capsize=5,
                color='green', linewidth=2)
    ax2.set_xlabel('Radius Percentile (%)')
    ax2.set_ylabel('In-Cloud Accuracy')
    ax2.set_title('In-Cloud Prediction Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # 3. Cloud coverage vs radius_pct
    ax3 = fig.add_subplot(gs[0, 2])
    means = grouped['cloud_coverage'].mean()
    stds = grouped['cloud_coverage'].std()
    ax3.errorbar(means.index, means.values, yerr=stds.values, marker='^', capsize=5,
                color='blue', linewidth=2)
    ax3.set_xlabel('Radius Percentile (%)')
    ax3.set_ylabel('Cloud Coverage')
    ax3.set_title('Fraction of Points in Cloud')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% coverage')
    ax3.legend()

    # 4. Error rate vs radius_pct
    ax4 = fig.add_subplot(gs[1, 0])
    means = grouped['in_cloud_error_rate'].mean()
    stds = grouped['in_cloud_error_rate'].std()
    ax4.errorbar(means.index, means.values, yerr=stds.values, marker='v', capsize=5,
                color='red', linewidth=2)
    ax4.set_xlabel('Radius Percentile (%)')
    ax4.set_ylabel('In-Cloud Error Rate')
    ax4.set_title('In-Cloud Prediction Error Rate')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])

    # 5. Trade-off plot: Accuracy vs Coverage
    ax5 = fig.add_subplot(gs[1, 1])
    for shot in results_df['shot'].unique():
        shot_data = results_df[results_df['shot'] == shot]
        ax5.scatter(shot_data['cloud_coverage'], shot_data['in_cloud_accuracy'],
                   alpha=0.6, label=f'Shot {int(shot)}')
    ax5.set_xlabel('Cloud Coverage')
    ax5.set_ylabel('In-Cloud Accuracy')
    ax5.set_title('Accuracy-Coverage Trade-off')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)

    # 6. Heatmap of optimization scores
    ax6 = fig.add_subplot(gs[1, 2])
    pivot = results_df.pivot_table(
        values='optimization_score',
        index='shot',
        columns='radius_pct',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=False, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax6, cbar_kws={'label': 'Score'})
    ax6.set_title('Optimization Scores Heatmap')
    ax6.set_xlabel('Radius Percentile (%)')
    ax6.set_ylabel('Shot ID')

    # 7. Best parameters performance by shot
    ax7 = fig.add_subplot(gs[2, 0])
    best_radius_pct = results_df.groupby('radius_pct')['optimization_score'].mean().idxmax()
    best_results = results_df[results_df['radius_pct'] == best_radius_pct]

    x = range(len(best_results))
    width = 0.35
    ax7.bar([i - width/2 for i in x], best_results['in_cloud_correct'],
            width, label='Correct in cloud', color='green', alpha=0.7)
    ax7.bar([i + width/2 for i in x], best_results['in_cloud_incorrect'],
            width, label='Incorrect in cloud', color='red', alpha=0.7)
    ax7.set_xticks(x)
    ax7.set_xticklabels([f"{int(s)}" for s in best_results['shot']], rotation=45)
    ax7.set_xlabel('Shot ID')
    ax7.set_ylabel('Number of Points')
    ax7.set_title(f'Best Performance (radius_pct={best_radius_pct:.1f}%)')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Distribution of distances for best parameters
    ax8 = fig.add_subplot(gs[2, 1])
    radius_values = results_df[results_df['radius_pct'] == best_radius_pct]['radius_value'].values
    ax8.hist(radius_values, bins=20, edgecolor='black', alpha=0.7)
    ax8.set_xlabel('Radius Value (standardized space)')
    ax8.set_ylabel('Frequency')
    ax8.set_title(f'Distribution of Radius Values at {best_radius_pct:.1f}%')
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Overall metrics summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # Calculate summary statistics for best parameters
    best_mean_score = best_results['optimization_score'].mean()
    best_std_score = best_results['optimization_score'].std()
    best_mean_acc = best_results['in_cloud_accuracy'].mean()
    best_mean_coverage = best_results['cloud_coverage'].mean()
    best_mean_error = best_results['in_cloud_error_rate'].mean()

    summary_text = f"""
    BEST HYPERPARAMETER SUMMARY
    {'='*35}

    Optimal radius_pct: {best_radius_pct:.1f}%

    Performance Metrics:
    • Optimization Score: {best_mean_score:.4f} ± {best_std_score:.4f}
    • In-Cloud Accuracy: {best_mean_acc:.3f}
    • Cloud Coverage: {best_mean_coverage:.3f}
    • In-Cloud Error Rate: {best_mean_error:.3f}

    Shot-by-Shot Results:
    """

    for _, row in best_results.iterrows():
        summary_text += f"\n    Shot {int(row['shot'])}: Score={row['optimization_score']:.3f}"

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top')

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f'hypersphere_optimization_{timestamp}.png')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {filename}")

    return fig

def save_results(results: List[Dict], best_radius_pct: float) -> pd.DataFrame:
    """Save optimization results to files"""

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(OUTPUT_DIR, f'hypersphere_optimization_results_{timestamp}.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to: {csv_filename}")

    # Calculate detailed statistics for best parameters
    best_results = results_df[results_df['radius_pct'] == best_radius_pct]

    # Save best parameters to JSON
    best_params = {
        'radius_pct': float(best_radius_pct),
        'test_shots': TEST_SHOTS,
        'timestamp': timestamp,
        'performance_summary': {
            'mean_optimization_score': float(best_results['optimization_score'].mean()),
            'std_optimization_score': float(best_results['optimization_score'].std()),
            'mean_in_cloud_accuracy': float(best_results['in_cloud_accuracy'].mean()),
            'mean_cloud_coverage': float(best_results['cloud_coverage'].mean()),
            'mean_in_cloud_error_rate': float(best_results['in_cloud_error_rate'].mean()),
        },
        'feature_columns': FEATURE_COLUMNS
    }

    json_filename = os.path.join(OUTPUT_DIR, f'best_hypersphere_params_{timestamp}.json')
    with open(json_filename, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best parameters saved to: {json_filename}")

    # Also save a "latest" version for easy access
    latest_json = os.path.join(OUTPUT_DIR, 'best_hypersphere_params_latest.json')
    with open(latest_json, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Latest parameters saved to: {latest_json}")

    return results_df

def print_detailed_summary(results_df: pd.DataFrame, best_radius_pct: float):
    """Print detailed summary statistics of the optimization"""

    print("\n" + "="*80)
    print("DETAILED OPTIMIZATION SUMMARY")
    print("="*80)

    # Best parameters analysis
    best_results = results_df[results_df['radius_pct'] == best_radius_pct]

    print(f"\n1. OPTIMAL HYPERPARAMETER")
    print(f"   radius_pct: {best_radius_pct:.1f}%")

    print(f"\n2. PERFORMANCE METRICS (at optimal radius_pct)")
    print(f"   Average optimization score: {best_results['optimization_score'].mean():.4f} ± {best_results['optimization_score'].std():.4f}")
    print(f"   Average in-cloud accuracy: {best_results['in_cloud_accuracy'].mean():.3f} ± {best_results['in_cloud_accuracy'].std():.3f}")
    print(f"   Average cloud coverage: {best_results['cloud_coverage'].mean():.3f} ± {best_results['cloud_coverage'].std():.3f}")
    print(f"   Average in-cloud error rate: {best_results['in_cloud_error_rate'].mean():.3f} ± {best_results['in_cloud_error_rate'].std():.3f}")
    print(f"   Average overall accuracy: {best_results['overall_accuracy'].mean():.3f} ± {best_results['overall_accuracy'].std():.3f}")

    print(f"\n3. SHOT-BY-SHOT BREAKDOWN")
    for _, row in best_results.iterrows():
        print(f"   Shot {int(row['shot'])}:")
        print(f"     - Score: {row['optimization_score']:.4f}")
        print(f"     - In-cloud: {int(row['in_cloud_correct'])} correct, {int(row['in_cloud_incorrect'])} incorrect")
        print(f"     - Out-cloud: {int(row['out_cloud_correct'])} correct, {int(row['out_cloud_incorrect'])} incorrect")
        print(f"     - Coverage: {row['cloud_coverage']:.3f} ({int(row['in_cloud_points'])}/{int(row['total_points'])} points)")

    # Compare to middle baseline (radius_pct = 95.0)
    baseline = results_df[results_df['radius_pct'] == 95.0]
    if not baseline.empty and best_radius_pct != 95.0:
        print(f"\n4. IMPROVEMENT OVER BASELINE (radius_pct=95.0%)")
        baseline_score = baseline['optimization_score'].mean()
        best_score = best_results['optimization_score'].mean()
        improvement = ((best_score - baseline_score) / abs(baseline_score)) * 100 if baseline_score != 0 else 0
        print(f"   Baseline score: {baseline_score:.4f}")
        print(f"   Optimized score: {best_score:.4f}")
        print(f"   Relative improvement: {improvement:+.1f}%")

    # Find the trade-off point
    print(f"\n5. PARAMETER SENSITIVITY ANALYSIS")
    grouped = results_df.groupby('radius_pct')

    # Find radius_pct with highest accuracy (might differ from best score)
    best_acc_radius = grouped['in_cloud_accuracy'].mean().idxmax()
    print(f"   Best accuracy achieved at: radius_pct={best_acc_radius:.1f}% (acc={grouped['in_cloud_accuracy'].mean()[best_acc_radius]:.3f})")

    # Find radius_pct with coverage closest to 50%
    coverage_means = grouped['cloud_coverage'].mean()
    best_coverage_radius = coverage_means.iloc[(coverage_means - 0.5).abs().argmin()]
    idx = (coverage_means - 0.5).abs().argmin()
    best_coverage_radius_pct = coverage_means.index[idx]
    print(f"   Most balanced coverage at: radius_pct={best_coverage_radius_pct:.1f}% (coverage={best_coverage_radius:.3f})")

def main():
    """Main function to run the hypersphere optimization"""

    try:
        # Run grid search optimization
        results, best_radius_pct = grid_search_optimization()

        # Save results
        results_df = save_results(results, best_radius_pct)

        # Create comprehensive visualization
        create_comprehensive_visualization(results_df)

        # Print detailed summary
        print_detailed_summary(results_df, best_radius_pct)

        print("\n" + "="*80)
        print("HYPERSPHERE OPTIMIZATION COMPLETE!")
        print(f"Results saved in: {OUTPUT_DIR}/")
        print("="*80)

        return results_df, best_radius_pct

    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    results_df, best_radius_pct = main()