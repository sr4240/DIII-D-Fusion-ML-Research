import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """
    Load and prepare the dataset from CSV
    """
    print("=== Loading Plasma Dataset from CSV ===")
    
    # Load the CSV dataset
    df = pd.read_csv('plasma_data.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()
    print(f"After removing shot 191675: {df.shape}")
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Add an index column to keep track of original row numbers
    df['original_index'] = df.index
    
    return df

def select_features_and_clean(df):
    """
    Select features and clean the dataset for 4-state classification
    """
    print("\n=== Feature Selection and Data Cleaning ===")
    
    # Select features based on the provided example
    selected_features = [
        'iln3iamp', 'betan', 'density', 'n_eped', 'li', 'tritop','fs04_max_smoothed'
    ]
    
    # Check which features are available
    available_features = [f for f in selected_features if f in df.columns]
    print(f"Available features: {available_features}")
    print(f"Features not found: {[f for f in selected_features if f not in df.columns]}")
    
    # Clean the dataframe - remove rows with missing values in selected features
    df_cleaned = df.dropna(subset=available_features, how='any')
    print(f"After removing rows with missing values: {len(df_cleaned)}")
    
    # Filter for n=3 and remove N/A states (0)
    df_cleaned = df_cleaned[df_cleaned['n'] == 3]
    df_cleaned = df_cleaned[df_cleaned['state'] != 0]
    print(f"After filtering n=3 and removing N/A states: {len(df_cleaned)}")
    
    # Keep original state for 4-state classification
    df_cleaned['state'] = df_cleaned['state']
    
    # Strictly keep only the selected features plus columns required for splitting/label
    columns_to_keep = ['shot', 'time', 'state'] + available_features
    df_cleaned = df_cleaned[columns_to_keep].copy()
    print(f"Columns kept for modeling (strict): {['shot', 'time', 'state'] + available_features}")
    
    # Check 4-state distribution
    state_counts = df_cleaned['state'].value_counts().sort_index()
    state_names = {1: 'Suppressed', 2: 'Dithering', 3: 'Mitigated', 4: 'ELMing'}
    
    print(f"\n4-state distribution:")
    for state, count in state_counts.items():
        if state in state_names:
            print(f"  {state_names[state]} (State {state}): {count:6d} records ({count/len(df_cleaned)*100:.1f}%)")
        else:
            print(f"  State {state}: {count:6d} records ({count/len(df_cleaned)*100:.1f}%)")
    
    return df_cleaned, available_features, state_names

def prepare_chronological_splits(df_cleaned, available_features):
    """
    Prepare training/testing data with an 80/20 shot-based split using a random
    permutation of shots (seed=42).
    """
    print("\n=== Data Splitting (80/20 by shot; random shot split seed=42) ===")

    # Random shot split for 80/20 split
    unique_shots = df_cleaned['shot'].unique()
    num_shots = len(unique_shots)
    np.random.seed(43)
    shuffled_shots = np.random.permutation(unique_shots)
    train_count = int(np.floor(0.80 * num_shots))
    train_shots = shuffled_shots[:train_count]
    test_shots = shuffled_shots[train_count:]

    print(f"Total shots: {num_shots} | Train shots: {len(train_shots)} | Test shots: {len(test_shots)}")

    # Prepare input (X) and target (y)
    train_df = df_cleaned[df_cleaned['shot'].isin(train_shots)]
    test_df = df_cleaned[df_cleaned['shot'].isin(test_shots)]

    X_train = train_df[available_features]
    y_train = train_df['state']
    X_test = test_df[available_features]
    y_test = test_df['state']

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier for 4-state classification
    """
    print("\n=== Training Random Forest (4-State Classification) ===")
    
    # Train a Random Forest Classifier with parameters similar to Random_Split.py
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=35,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=43,
        n_jobs=-1  # Use all available cores
    )
    
    clf.fit(X_train, y_train)
    
    print("Random Forest training completed!")
    
    return clf

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained model
    """
    print("\n=== Model Evaluation ===")
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (last 20% data): {accuracy:.4f}")
    
    # Evaluate on training set
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Check for overfitting
    overfitting = train_accuracy - accuracy
    print(f"Overfitting (Train - Test accuracy): {overfitting:.4f}")
    
    return y_pred, y_train_pred

def plot_confusion_matrix(y_test, y_pred, state_names):
    """
    Plot confusion matrix with accuracy in title - matching Random_Split.py style
    """
    print("\n=== Confusion Matrix ===")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion Matrix (Counts) - matching Random_Split.py style
    plt.figure(figsize=(7, 6))
    cm_counts = confusion_matrix(y_test, y_pred)
    class_names = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Counts) - Accuracy: {accuracy:.4f}')
    plt.savefig('confusion_matrix_enhanced.png', dpi=600, bbox_inches='tight')
    print("Enhanced confusion matrix saved to: confusion_matrix_enhanced.png")
    plt.show()


def print_classification_report(y_test, y_pred, state_names):
    """
    Print detailed classification report - matching Random_Split.py style
    """
    print("\n=== Classification Report ===")
    
    # Classification Report - matching Random_Split.py style
    print(classification_report(y_test, y_pred, target_names=['Suppressed (1)', 'Dithering (2)', 'Mitigated (3)', 'ELMing (4)']))
    
    # Add detailed prediction analysis for each state - matching Random_Split.py
    print("\nDetailed Prediction Analysis by State:")
    for state, state_name in [(1, 'Suppressed'), (2, 'Dithering'), (3, 'Mitigated'), (4, 'ELMing')]:
        state_mask = y_test == state
        if state_mask.sum() > 0:
            correct_predictions = (y_pred[state_mask] == state).sum()
            total_predictions = state_mask.sum()
            state_accuracy = correct_predictions / total_predictions
            print(f"{state_name} (State {state}): {correct_predictions}/{total_predictions} correct ({state_accuracy:.3f})")

def analyze_feature_importance(clf, available_features):
    """
    Analyze and plot feature importance
    """
    print("\n=== Feature Importance Analysis ===")
    
    # Get feature importance
    feature_importances = pd.Series(clf.feature_importances_, index=available_features).sort_values(ascending=False)
    
    print("Feature Importances:")
    print(feature_importances)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importances)))
    bars = plt.barh(range(len(feature_importances)), feature_importances.values, color=colors)
    plt.yticks(range(len(feature_importances)), feature_importances.index)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importances for Binary Plasma State Classification', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 most important features:")
    for i, (feature, importance) in enumerate(feature_importances.head(10).items(), 1):
        print(f"{i:2d}. {feature}: {importance:.4f}")

def print_model_parameters(clf):
    """
    Print the Random Forest parameters
    """
    print(f"\n=== Random Forest Parameters ===")
    print(f"n_estimators: {clf.n_estimators}")
    print(f"max_depth: {clf.max_depth}")
    print(f"min_samples_split: {clf.min_samples_split}")
    print(f"min_samples_leaf: {clf.min_samples_leaf}")
    print(f"max_features: {clf.max_features}")
    print(f"random_state: {clf.random_state}")

def main():
    """
    Main execution function
    """
    print("=== 4-State Plasma State Classification with Random Forest ===\n")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Select features and clean data
    df_cleaned, available_features, state_names = select_features_and_clean(df)
    
    # Prepare chronological data splits
    X_train, X_test, y_train, y_test = prepare_chronological_splits(df_cleaned, available_features)
    
    # Train Random Forest
    clf = train_random_forest(X_train, y_train)
    
    # Evaluate model
    y_pred, y_train_pred = evaluate_model(clf, X_train, X_test, y_train, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, state_names)
    
    # Print classification report
    print_classification_report(y_test, y_pred, state_names)
    
    # Analyze feature importance
    analyze_feature_importance(clf, available_features)
    
    # Print model parameters
    print_model_parameters(clf)
    
    # Print summary statistics - matching Random_Split.py style
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSummary:")
    print(f"Final accuracy: {accuracy:.3f}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {len(available_features)}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main() 