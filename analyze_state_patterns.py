#!/usr/bin/env python3
"""
Analyze what patterns distinguish each plasma state
Shows what the classifier learns about each state
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Analyzing Plasma State Patterns")
print("=" * 60)

# Load sample data
print("\nLoading data...")
df = pd.read_csv('plasma_data.csv', nrows=50000)

# Clean data
if df.iloc[0, 2] == 'state':
    df = df.iloc[1:].reset_index(drop=True)

df['state'] = pd.to_numeric(df['state'], errors='coerce')

# Key features that distinguish states
key_features = ['fs04', 'betan', 'density', 'n', 'li', 'q95', 'Ip']
available_features = [f for f in key_features if f in df.columns]

X = df[available_features].fillna(df[available_features].median())
y = df['state'].values

# State names
state_names = {
    1: 'L-Mode',
    2: 'Dithering',
    3: 'H-Mode ELM-free',
    4: 'ELMing'
}

# Analyze feature distributions per state
print("\n" + "=" * 60)
print("Feature Statistics by State")
print("=" * 60)

for state in [1, 2, 3, 4]:
    mask = y == state
    print(f"\n{state_names[state]} (State {state}):")
    print("-" * 40)

    state_data = X[mask]

    for feature in available_features[:5]:  # Show top 5 features
        values = state_data[feature].values
        print(f"  {feature:10s}: mean={np.mean(values):8.3f}, std={np.std(values):8.3f}, "
              f"median={np.median(values):8.3f}")

# Train a simple classifier to get feature importance
print("\n" + "=" * 60)
print("Learning State Patterns")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_scaled, y)

# Get feature importance
importance = rf.feature_importances_
feature_importance = sorted(zip(available_features, importance),
                          key=lambda x: x[1], reverse=True)

print("\nFeature Importance (what distinguishes states):")
print("-" * 40)
for feature, imp in feature_importance:
    bar_width = int(40 * imp / max(importance))
    bar = '█' * bar_width + '░' * (40 - bar_width)
    print(f"  {feature:10s}: {bar} {imp:.4f}")

# Analyze decision rules
print("\n" + "=" * 60)
print("Key Patterns Learned by Classifier")
print("=" * 60)

# Get sample decision tree rules
from sklearn.tree import _tree

def get_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, depth, parent_rule=""):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Left child (<=)
            left_rule = f"{parent_rule} & " if parent_rule else ""
            left_rule += f"({name} <= {threshold:.3f})"
            recurse(tree_.children_left[node], depth + 1, left_rule)

            # Right child (>)
            right_rule = f"{parent_rule} & " if parent_rule else ""
            right_rule += f"({name} > {threshold:.3f})"
            recurse(tree_.children_right[node], depth + 1, right_rule)
        else:
            # Leaf node
            values = tree_.value[node][0]
            predicted_class = np.argmax(values)
            if predicted_class in [1, 2, 3, 4] and depth <= 3:  # Only shallow rules
                confidence = values[predicted_class-1] / np.sum(values)
                if confidence > 0.7:  # High confidence rules
                    rules.append((parent_rule, predicted_class, confidence))

    recurse(0, 0)
    return rules

# Get rules from first tree
if hasattr(rf, 'estimators_'):
    tree = rf.estimators_[0]
    rules = get_rules(tree, available_features)

    # Show top rules for each state
    for state in [1, 2, 3, 4]:
        state_rules = [(r, c, conf) for r, c, conf in rules if c == state]
        if state_rules:
            print(f"\n{state_names[state]} patterns (sample rules):")
            print("-" * 40)
            for rule, _, conf in state_rules[:2]:  # Top 2 rules
                print(f"  If {rule}")
                print(f"    → {state_names[state]} (confidence: {conf:.2%})")

# Show distinguishing thresholds
print("\n" + "=" * 60)
print("Distinguishing Thresholds")
print("=" * 60)

for feature in available_features[:3]:  # Top 3 features
    print(f"\n{feature}:")
    for state in [1, 2, 3, 4]:
        mask = y == state
        values = X[mask][feature].values
        p25, p50, p75 = np.percentile(values, [25, 50, 75])
        print(f"  {state_names[state]:15s}: 25%={p25:6.2f}, median={p50:6.2f}, 75%={p75:6.2f}")

# Physical interpretation
print("\n" + "=" * 60)
print("Physical Interpretation")
print("=" * 60)

print("""
The classifier learns these patterns from the data:

1. L-Mode (State 1):
   - Lower density and pressure
   - Lower fs04 activity (no ELMs)
   - More stable plasma parameters

2. Dithering (State 2):
   - Intermediate values between L and H mode
   - Oscillating parameters
   - Transitional behavior

3. H-Mode ELM-free (State 3):
   - Higher density and pressure than L-mode
   - Low fs04 activity (suppressed ELMs)
   - Higher confinement

4. ELMing (State 4):
   - High fs04 spikes (edge instabilities)
   - H-mode pressure/density
   - Periodic edge losses

The classifier doesn't "understand" physics - it just finds
statistical patterns that correlate with the labeled states.
""")

print("=" * 60)