import argparse
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
	feature_df = df
	missing = [c for c in FEATURE_COLUMNS if c not in feature_df.columns]
	if missing:
		raise ValueError(f"Missing required feature columns: {missing}")
	feature_df = feature_df[FEATURE_COLUMNS]
	feature_df = feature_df.select_dtypes(include=[np.number])
	feature_cols = list(feature_df.columns)
	return feature_df, feature_cols


def fit_imputer_and_standardizer(features: pd.DataFrame) -> Tuple[SimpleImputer, StandardScaler, np.ndarray]:
	imputer = SimpleImputer(strategy="median")
	X_imputed = imputer.fit_transform(features.values)
	scaler = StandardScaler()
	X_std = scaler.fit_transform(X_imputed)
	return imputer, scaler, X_std


def percentile_radius(center: np.ndarray, X: np.ndarray, radius_pct: float) -> Tuple[float, np.ndarray]:
	d = np.linalg.norm(X - center[None, :], axis=1)
	r = float(np.percentile(d, radius_pct))
	return r, d


def geometric_median(X: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
	"""Compute the geometric median using Weiszfeld's algorithm (deterministic)."""
	y = np.median(X, axis=0)
	for _ in range(max_iter):
		diffs = X - y[None, :]
		dists = np.linalg.norm(diffs, axis=1)
		# If y coincides with a data point, that's the geometric median
		if np.any(dists == 0.0):
			return y
		weights = 1.0 / np.maximum(dists, 1e-12)
		y_new = np.sum(X * weights[:, None], axis=0) / np.sum(weights)
		if np.linalg.norm(y_new - y) < tol:
			return y_new
		y = y_new
	return y


def compute_fractional_meb(
	X_std: np.ndarray,
	radius_pct: float,
) -> Tuple[np.ndarray, float]:
	"""
	Deterministic fractional minimum enclosing ball using ONLY the geometric median
	as center. The radius is the given percentile of Euclidean distances to that center.
	"""
	if X_std.shape[0] == 0:
		raise ValueError("Empty feature matrix")

	center_geom = geometric_median(X_std)
	radius, _ = percentile_radius(center_geom, X_std, radius_pct)
	return center_geom, radius


def evaluate_shot_inside_fraction(
	df: pd.DataFrame,
	shot_id: int,
	imputer: SimpleImputer,
	scaler: StandardScaler,
	feature_cols: List[str],
	center: np.ndarray,
	radius: float,
) -> Tuple[float, int, int]:
	shot_df = df[df["shot"] == shot_id]
	if shot_df.empty:
		raise ValueError(f"Shot {shot_id} not found in CSV")

	X_shot = shot_df[feature_cols].values
	X_shot_imputed = imputer.transform(X_shot)
	X_shot_std = scaler.transform(X_shot_imputed)
	d = np.linalg.norm(X_shot_std - center[None, :], axis=1)
	inside = (d <= radius)
	n_inside = int(inside.sum())
	n_total = int(inside.shape[0])
	percent_inside = 100.0 * n_inside / float(n_total)
	return percent_inside, n_inside, n_total



def parse_args(argv: List[str] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Fractional Minimum Enclosing Ball (hypersphere) for plasma data")
	parser.add_argument("--csv", default="/mnt/homes/sr4240/my_folder/plasma_data.csv", help="Path to plasma_data.csv")
	parser.add_argument("--shot", type=int, default=169472, help="Shot ID to evaluate percent inside")
	parser.add_argument("--radius-pct", type=float, default=95.0, help="Percentile coverage for the ball (default: 95)")
	return parser.parse_args(argv)


def main(argv: List[str] = None) -> int:
	args = parse_args(argv)

	df = load_csv(args.csv)
	features_df, feature_cols = select_features(df)
	if features_df.empty:
		raise ValueError("No numeric feature columns available after dropping")

	imputer, scaler, X_std = fit_imputer_and_standardizer(features_df)
	center, radius = compute_fractional_meb(
		X_std=X_std,
		radius_pct=args.radius_pct,
	)

	print("Cloud (fractional MEB) parameters:")
	print(f"- features: {feature_cols}")
	print(f"- radius percentile: {args.radius_pct:.1f}")
	print(f"- radius (standardized space): {radius:.6f}")

	percent_inside, n_inside, n_total = evaluate_shot_inside_fraction(
		df=df,
		shot_id=args.shot,
		imputer=imputer,
		scaler=scaler,
		feature_cols=feature_cols,
		center=center,
		radius=radius,
	)
	print("")
	print(f"Shot {args.shot} inside-ball fraction: {percent_inside:.2f}% ({n_inside}/{n_total})")
	print(f"Shot {args.shot} outside-ball fraction: {100.0 - percent_inside:.2f}% ({n_total - n_inside}/{n_total})")

	return 0


if __name__ == "__main__":
	sys.exit(main())


