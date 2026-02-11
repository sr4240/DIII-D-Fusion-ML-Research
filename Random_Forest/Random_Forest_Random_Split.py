import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

plt.close('all')

# Load the plasma data from CSV file
df = pd.read_csv('plasma_data.csv')

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Add an index column to keep track of original row numbers
df['original_index'] = df.index

# Select features and the target variable
selected_features = ['iln3iamp', 'betan', 'density', 'n_eped', 'li', 'tritop','fs04_max_smoothed'
]
target_column = 'state'

# Clean the dataframe
df_cleaned = df.dropna(subset=selected_features, how='any')
df_cleaned = df_cleaned[df_cleaned['n'] == 3]

# Remove rows where state is 'N/A' (assuming N/A is not in your 4 states: 1,2,3,4)
# If you don't have N/A states, you can comment out this line
# df_cleaned = df_cleaned[df_cleaned['state'] != 0]

# Prepare input (X) and target (y)
X = df_cleaned[selected_features]
y = df_cleaned[target_column]

# Split into training and testing sets
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, df_cleaned['original_index'], test_size=0.2, random_state=43)

# Store test data if needed (remove this line if not required)
# OMFIT['X_test'] = X_test.copy()
# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=500, max_depth=35, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=43, n_jobs=-1)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix (Counts)
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

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Suppressed (1)', 'Dithering (2)', 'Mitigated (3)', 'ELMing (4)'], digits=4))

# Add the predictions to the DataFrame
df_test = X_test.copy()
df_test['actual_state'] = y_test
df_test['pred_state'] = y_pred

# Create a column to classify the prediction results for 4-state classification
df_test['condition'] = np.where(df_test['pred_state'] == df_test['actual_state'], 'Correct',
                                'Incorrect')

# Add detailed prediction analysis for each state
print("\nDetailed Prediction Analysis by State:")
for state, state_name in [(1, 'Suppressed'), (2, 'Dithering'), (3, 'Mitigated'), (4, 'ELMing')]:
    state_mask = df_test['actual_state'] == state
    if state_mask.sum() > 0:
        correct_predictions = (df_test.loc[state_mask, 'pred_state'] == state).sum()
        total_predictions = state_mask.sum()
        state_accuracy = correct_predictions / total_predictions
        print(f"{state_name} (State {state}): {correct_predictions}/{total_predictions} correct ({state_accuracy:.3f})")

# Print summary statistics
print(f"\nSummary:")
print(f"Final accuracy: {accuracy:.3f}")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Number of features: {len(selected_features)}")
