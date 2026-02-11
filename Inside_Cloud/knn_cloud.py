import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: List[str] = [
	"iln3iamp",
	"betan",
	"density",
	"n_eped",
	"li",
	"tritop",
	"fs04_max_smoothed",
]


def load_csv(csv_path: str) -> pd.DataFrame:
	return pd.read_csv(csv_path)


def select_features(df: pd.DataFrame, drop_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
	if drop_columns is None:
		drop_columns = []

	# Identify label if present (not used here but returned for parity)
	label = df["state"] if ("state" in df.columns and "state" not in drop_columns) else None
	if "state" in df.columns and "state" not in drop_columns:
		drop_columns.append("state")

	feature_df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")

	missing = [c for c in FEATURE_COLUMNS if c not in feature_df.columns]
	if missing:
		raise ValueError(f"Missing required feature columns: {missing}")

	feature_df = feature_df[FEATURE_COLUMNS]
	feature_df = feature_df.select_dtypes(include=[np.number])
	feature_cols = list(feature_df.columns)
	return feature_df, label, feature_cols


def fit_imputer_and_standardizer(features: pd.DataFrame) -> Tuple[SimpleImputer, StandardScaler, np.ndarray]:
	imputer = SimpleImputer(strategy="median")
	X_imputed = imputer.fit_transform(features.values)
	scaler = StandardScaler()
	X_std = scaler.fit_transform(X_imputed)
	return imputer, scaler, X_std


def compute_mean_neighbor_distances(X_std: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
	"""
	For each point, compute the mean Euclidean distance to its n_neighbors nearest neighbors
	(excluding itself). Returns an array of shape (n_samples,).
	"""
	if X_std.shape[0] <= n_neighbors:
		raise ValueError("Number of neighbors must be less than number of samples")

	# Fit on the training set
	nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
	nn.fit(X_std)
	distances, indices = nn.kneighbors(X_std, return_distance=True)
	# distances[:, 0] corresponds to distance to self (0); drop it
	mean_dists = distances[:, 1:].mean(axis=1)
	return mean_dists


def transform_new_shot(
	shot_df: pd.DataFrame,
	feature_cols: List[str],
	imputer: SimpleImputer,
	scaler: StandardScaler,
) -> np.ndarray:
	X = shot_df[feature_cols].values
	X_imputed = imputer.transform(X)
	X_std = scaler.transform(X_imputed)
	return X_std


def compute_mean_distances_to_training_neighbors(
	X_train_std: np.ndarray,
	X_eval_std: np.ndarray,
	n_neighbors: int = 20,
) -> np.ndarray:
	"""
	For each evaluation point, compute mean distance to its n_neighbors nearest neighbors
	in the training set.
	"""
	if X_train_std.shape[0] <= n_neighbors:
		raise ValueError("Number of neighbors must be less than number of training samples")

	nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
	nn.fit(X_train_std)
	distances, _ = nn.kneighbors(X_eval_std, return_distance=True)
	return distances.mean(axis=1)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="kNN cloud thresholding and shot evaluation")
	parser.add_argument("--csv", default="/mnt/homes/sr4240/my_folder/plasma_data.csv", help="Path to CSV containing training data and other shots")
	parser.add_argument("--heldout-shot", type=int, default=169501, help="Shot ID to evaluate (held-out)")
	parser.add_argument("--neighbors", type=int, default=20, help="Number of nearest neighbors to average")
	parser.add_argument("--drop-cols", nargs="*", default=["shot", "time"], help="Additional columns to drop from features")
	parser.add_argument("--percentile", type=float, default=95.0, help="Percentile over training mean distances")
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)

	df = load_csv(args.csv)

	# Exclude held-out shot from training data
	train_df = df[df["shot"] != args.heldout_shot]
	features_df, _labels, feature_cols = select_features(train_df, drop_columns=args.drop_cols)

	if features_df.empty:
		raise ValueError("No numeric feature columns available after dropping")

	imputer, scaler, X_train_std = fit_imputer_and_standardizer(features_df)

	# Compute per-point mean distance to 20-NN among training points
	train_mean_dists = compute_mean_neighbor_distances(X_train_std, n_neighbors=args.neighbors)
	threshold = float(np.percentile(train_mean_dists, args.percentile))

	print("Training set neighbor-distance summary:")
	print(f"- neighbors: {args.neighbors}")
	print(f"- percentile: {args.percentile:.1f}")
	print(f"- threshold (mean distance at {args.percentile:.1f}th pct): {threshold:.6f}")

	# Evaluate held-out shot
	shot_df = df[df["shot"] == args.heldout_shot]
	if shot_df.empty:
		raise ValueError(f"Held-out shot {args.heldout_shot} not found in CSV")

	X_eval_std = transform_new_shot(shot_df, feature_cols, imputer, scaler)
	eval_mean_dists = compute_mean_distances_to_training_neighbors(
		X_train_std, X_eval_std, n_neighbors=args.neighbors
	)

	below = int((eval_mean_dists <= threshold).sum())
	above = int((eval_mean_dists > threshold).sum())
	total = int(eval_mean_dists.shape[0])
	pct_below = 100.0 * below / float(total)
	pct_above = 100.0 * above / float(total)

	print("")
	print(f"Held-out shot {args.heldout_shot} evaluation:")
	print(f"- below-or-equal threshold: {pct_below:.2f}% ({below}/{total})")
	print(f"- above threshold: {pct_above:.2f}% ({above}/{total})")

	return 0


if __name__ == "__main__":
	sys.exit(main())
