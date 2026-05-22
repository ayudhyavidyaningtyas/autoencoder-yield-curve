#!/usr/bin/env python3
"""Replicate a shallow autoencoder factor model for the JGB yield curve.

The paper uses TensorFlow. This coursework implementation therefore trains the
same 6-3-6 tanh/linear architecture with Keras while keeping preprocessing,
diagnostics, and output tables explicit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig")
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATURITIES = ["2Y", "5Y", "7Y", "10Y", "15Y", "20Y"]
MATURITY_YEARS = np.array([2, 5, 7, 10, 15, 20], dtype=float)
PROXY_NAMES = ["Level", "Slope 20Y-2Y", "Curvature 2*10Y-2Y-20Y"]


@dataclass
class AutoencoderFit:
    autoencoder: Any
    encoder: Any
    w_encoder: np.ndarray
    w_decoder: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    scale: str
    batch_size: int
    loss_history: List[float]
    seed: int


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Replicate the JGB yield-curve autoencoder paper."
    )
    parser.add_argument("--csv", type=Path, default=root / "data" / "jgbcme_all.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--start-date", default="1992-07-01")
    parser.add_argument("--end-date", default="2019-07-31")
    parser.add_argument("--hidden", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Keras batch size. Use 0 for full-batch training on each sample window.",
    )
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale",
        choices=["none", "standardize"],
        default="none",
        help="Use raw yields by default to match the paper; standardize for a numerical sensitivity check.",
    )
    parser.add_argument("--skip-trading", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--trading-epochs", type=int, default=1800)
    parser.add_argument("--hold-weeks", type=int, default=4)
    parser.add_argument(
        "--run-forecast-models",
        action="store_true",
        help="Also run the paper-style LSTM and VAR one-month-ahead trading comparison.",
    )
    parser.add_argument("--forecast-lags", type=int, default=4)
    parser.add_argument("--lstm-units", type=int, default=8)
    parser.add_argument("--lstm-epochs", type=int, default=80)
    parser.add_argument("--lstm-learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--forecast-batch-size",
        type=int,
        default=0,
        help="Keras LSTM batch size. Use 0 for full-batch training in each rolling window.",
    )
    return parser.parse_args()


def load_jgb_yields(
    csv_path: Path,
    start_date: str,
    end_date: str,
    maturities: Iterable[str] = MATURITIES,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Download jgbcme_all.csv from the MOF historical "
            "interest-rate page and place it in data/."
        )

    raw = pd.read_csv(csv_path, skiprows=1, na_values=["-", " -", ""])
    raw["Date"] = pd.to_datetime(raw["Date"])
    for col in raw.columns:
        if col != "Date":
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    daily = raw.set_index("Date").sort_index()
    daily = daily.loc[start_date:end_date, list(maturities)].dropna(how="any")

    weekly = daily.resample("W-FRI").last().dropna(how="any")
    weekly.index.name = "Date"
    return weekly


def run_pca(yields: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    x = yields.to_numpy(dtype=float)
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2 / (len(x) - 1)
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)

    pca_table = pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance": explained,
            "cumulative_variance": cumulative,
        }
    )
    loadings = pd.DataFrame(
        vt[:3],
        columns=yields.columns,
        index=["PC1", "PC2", "PC3"],
    )

    # Stabilize signs for readable level/slope/curvature plots.
    if loadings.loc["PC1"].mean() < 0:
        loadings.loc["PC1"] *= -1
    if loadings.loc["PC2", "20Y"] < loadings.loc["PC2", "2Y"]:
        loadings.loc["PC2"] *= -1
    return pca_table, loadings, vt


def pca_reconstruction(
    yields: pd.DataFrame,
    components: int = 3,
) -> pd.DataFrame:
    x = yields.to_numpy(dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    reconstructed = mean + (u[:, :components] * singular_values[:components]) @ vt[:components]
    return pd.DataFrame(reconstructed, index=yields.index, columns=yields.columns)


def prepare_training_data(
    x: np.ndarray, scale: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scale == "none":
        mean = np.zeros((1, x.shape[1]), dtype=float)
        std = np.ones((1, x.shape[1]), dtype=float)
        return x.copy(), mean, std

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def import_keras() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the autoencoder. Install it in a "
            "TensorFlow-supported Python environment, for example with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    tf.get_logger().setLevel("ERROR")
    return tf


def train_autoencoder_once(
    x: np.ndarray,
    hidden: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    scale: str,
    batch_size: int,
) -> AutoencoderFit:
    tf = import_keras()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    x_scaled, mean, std = prepare_training_data(x, scale)
    x_scaled = x_scaled.astype(np.float32)
    n_obs, n_features = x_scaled.shape
    effective_batch_size = n_obs if batch_size <= 0 else min(batch_size, n_obs)

    inputs = tf.keras.Input(shape=(n_features,), name="yield_curve")
    factors = tf.keras.layers.Dense(
        hidden,
        activation="tanh",
        use_bias=False,
        name="factors",
    )(inputs)
    outputs = tf.keras.layers.Dense(
        n_features,
        activation="linear",
        use_bias=False,
        name="reconstructed_yields",
    )(factors)

    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="yield_curve_autoencoder")
    encoder = tf.keras.Model(inputs=inputs, outputs=factors, name="yield_curve_encoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    history = autoencoder.fit(
        x_scaled,
        x_scaled,
        epochs=epochs,
        batch_size=effective_batch_size,
        shuffle=False,
        verbose=0,
    )
    losses = [float(value) for value in history.history["loss"]]
    w_encoder = autoencoder.get_layer("factors").get_weights()[0]
    w_decoder = autoencoder.get_layer("reconstructed_yields").get_weights()[0]

    return AutoencoderFit(
        autoencoder=autoencoder,
        encoder=encoder,
        w_encoder=w_encoder,
        w_decoder=w_decoder,
        train_mean=mean.ravel(),
        train_std=std.ravel(),
        scale=scale,
        batch_size=effective_batch_size,
        loss_history=losses,
        seed=seed,
    )


def train_autoencoder(
    x: np.ndarray,
    hidden: int,
    epochs: int,
    learning_rate: float,
    restarts: int,
    seed: int,
    scale: str,
    batch_size: int,
) -> AutoencoderFit:
    best: AutoencoderFit | None = None
    for i in range(restarts):
        fit = train_autoencoder_once(
            x=x,
            hidden=hidden,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed + i,
            scale=scale,
            batch_size=batch_size,
        )
        if best is None or fit.loss_history[-1] < best.loss_history[-1]:
            best = fit
    assert best is not None
    return best


def encode(fit: AutoencoderFit, x: np.ndarray) -> np.ndarray:
    x_scaled = (x - fit.train_mean) / fit.train_std
    return fit.encoder(x_scaled.astype(np.float32), training=False).numpy()


def reconstruct(fit: AutoencoderFit, x: np.ndarray) -> np.ndarray:
    x_scaled = (x - fit.train_mean) / fit.train_std
    x_hat_scaled = fit.autoencoder(x_scaled.astype(np.float32), training=False).numpy()
    return x_hat_scaled * fit.train_std + fit.train_mean


def make_proxies(yields: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Level": yields.mean(axis=1),
            "Slope 20Y-2Y": yields["20Y"] - yields["2Y"],
            "Curvature 2*10Y-2Y-20Y": 2.0 * yields["10Y"] - yields["2Y"] - yields["20Y"],
        },
        index=yields.index,
    )


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)


def correlation_table(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    corr = pd.DataFrame(index=left.columns, columns=right.columns, dtype=float)
    for left_col in left.columns:
        for right_col in right.columns:
            corr.loc[left_col, right_col] = left[left_col].corr(right[right_col])
    return corr


def align_hidden_factors(
    hidden_values: np.ndarray,
    proxies: pd.DataFrame,
    w_decoder: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hidden_df = pd.DataFrame(
        hidden_values,
        index=proxies.index,
        columns=[f"Node {i + 1}" for i in range(hidden_values.shape[1])],
    )
    corr = correlation_table(hidden_df, proxies)

    best_assignment = None
    best_score = -np.inf
    proxy_order = list(proxies.columns)
    for node_order in itertools.permutations(hidden_df.columns, len(proxy_order)):
        score = sum(abs(corr.loc[node, proxy]) for node, proxy in zip(node_order, proxy_order))
        if score > best_score:
            best_score = score
            best_assignment = dict(zip(node_order, proxy_order))

    assert best_assignment is not None
    labels = best_assignment
    signs: Dict[str, float] = {}
    for node, proxy in labels.items():
        signs[node] = 1.0 if corr.loc[node, proxy] >= 0 else -1.0

    ordered_nodes = [node for node, _ in sorted(labels.items(), key=lambda item: PROXY_NAMES.index(item[1]))]
    aligned_hidden = pd.DataFrame(index=hidden_df.index)
    aligned_decoder_rows = []
    for node in ordered_nodes:
        label = labels[node]
        sign = signs[node]
        aligned_hidden[label] = sign * hidden_df[node]
        aligned_decoder_rows.append(sign * w_decoder[int(node.split()[1]) - 1])

    aligned_decoder = pd.DataFrame(
        aligned_decoder_rows,
        index=[labels[node] for node in ordered_nodes],
        columns=MATURITIES,
    )
    return aligned_hidden, aligned_decoder, corr


def plot_yield_history(yields: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    yields.plot(ax=ax, linewidth=1.1)
    ax.set_title("JGB Yield History: Weekly Observations")
    ax.set_ylabel("Yield (%)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_pca(pca_table: pd.DataFrame, loadings: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(
        pca_table["component"],
        pca_table["cumulative_variance"],
        marker="o",
        color="#2f6f9f",
    )
    ax.axhline(0.99, color="#8a3ffc", linewidth=1.0, linestyle="--")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(pca_table["component"])
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("PCA Variance Explained")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for pc in loadings.index:
        ax.plot(MATURITY_YEARS, loadings.loc[pc], marker="o", label=pc)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(MATURITY_YEARS)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Loading")
    ax.set_title("PCA Loading Patterns")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_loadings.png", dpi=180)
    plt.close(fig)


def plot_autoencoder_results(
    yields: pd.DataFrame,
    reconstructed: pd.DataFrame,
    decoder: pd.DataFrame,
    hidden_aligned: pd.DataFrame,
    proxies: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label in decoder.index:
        ax.plot(MATURITY_YEARS, decoder.loc[label], marker="o", label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(MATURITY_YEARS)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Decoder weight")
    ax.set_title("Autoencoder Decoder Loadings")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "autoencoder_decoder_loadings.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for ax, label in zip(axes, PROXY_NAMES):
        comparison = pd.DataFrame(
            {
                f"Hidden: {label}": zscore(hidden_aligned[[label]])[label],
                f"Proxy: {label}": zscore(proxies[[label]])[label],
            }
        )
        comparison.plot(ax=ax, linewidth=1.0)
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_title("Hidden Factors vs Financial Proxies")
    axes[-1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "autoencoder_factor_proxies.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    for ax, maturity in zip(axes, ["2Y", "10Y", "20Y"]):
        ax.plot(yields.index, yields[maturity], label=f"Actual {maturity}", linewidth=1.0)
        ax.plot(
            reconstructed.index,
            reconstructed[maturity],
            label=f"Reconstructed {maturity}",
            linewidth=1.0,
            linestyle="--",
        )
        ax.set_ylabel("Yield (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_title("Actual vs Autoencoder-Reconstructed Yields")
    axes[-1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_fit.png", dpi=180)
    plt.close(fig)


def reconstruction_metrics(yields: pd.DataFrame, reconstructed: pd.DataFrame) -> pd.DataFrame:
    error = reconstructed - yields
    return pd.DataFrame(
        {
            "rmse_bp": np.sqrt((error**2).mean()) * 100.0,
            "mae_bp": error.abs().mean() * 100.0,
        }
    )


def summarize_reconstruction_metrics(
    label: str,
    actual: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> Dict[str, float | str]:
    metrics = reconstruction_metrics(actual, reconstructed)
    return {
        "model": label,
        "mean_rmse_bp": float(metrics["rmse_bp"].mean()),
        "max_rmse_bp": float(metrics["rmse_bp"].max()),
        "mean_mae_bp": float(metrics["mae_bp"].mean()),
        "max_mae_bp": float(metrics["mae_bp"].max()),
    }


def run_temporal_validation(
    yields: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    split_index = int(len(yields) * 0.8)
    train = yields.iloc[:split_index]
    test = yields.iloc[split_index:]
    validation_epochs = max(1200, args.epochs // 2)
    validation_restarts = max(3, args.restarts // 2)

    fit = train_autoencoder(
        train.to_numpy(dtype=float),
        hidden=args.hidden,
        epochs=validation_epochs,
        learning_rate=args.learning_rate,
        restarts=validation_restarts,
        seed=args.seed + 10_000,
        scale=args.scale,
        batch_size=args.batch_size,
    )

    train_reconstructed = pd.DataFrame(
        reconstruct(fit, train.to_numpy(dtype=float)),
        index=train.index,
        columns=train.columns,
    )
    test_reconstructed = pd.DataFrame(
        reconstruct(fit, test.to_numpy(dtype=float)),
        index=test.index,
        columns=test.columns,
    )

    validation = pd.DataFrame(
        [
            {
                **summarize_reconstruction_metrics(
                    "Autoencoder train window", train, train_reconstructed
                ),
                "period_start": str(train.index.min().date()),
                "period_end": str(train.index.max().date()),
                "observations": len(train),
                "epochs": validation_epochs,
                "restarts": validation_restarts,
                "scale": args.scale,
            },
            {
                **summarize_reconstruction_metrics(
                    "Autoencoder temporal holdout", test, test_reconstructed
                ),
                "period_start": str(test.index.min().date()),
                "period_end": str(test.index.max().date()),
                "observations": len(test),
                "epochs": validation_epochs,
                "restarts": validation_restarts,
                "scale": args.scale,
            },
        ]
    )
    validation.to_csv(output_dir / "temporal_validation_metrics.csv", index=False)
    return validation


def run_robustness_checks(
    yields: pd.DataFrame,
    pca_three_reconstructed: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    rows.append(
        {
            **summarize_reconstruction_metrics(
                "PCA reconstruction", yields, pca_three_reconstructed
            ),
            "hidden_nodes": 3,
            "scale": "centered PCA",
            "epochs": 0,
            "restarts": 0,
            "best_seed": "",
        }
    )

    robustness_epochs = max(1200, args.epochs // 2)
    robustness_restarts = max(3, args.restarts // 2)
    proxies = make_proxies(yields)

    for hidden in [2, 3, 4]:
        fit = train_autoencoder(
            yields.to_numpy(dtype=float),
            hidden=hidden,
            epochs=robustness_epochs,
            learning_rate=args.learning_rate,
            restarts=robustness_restarts,
            seed=args.seed + 20_000 + hidden,
            scale=args.scale,
            batch_size=args.batch_size,
        )
        reconstructed = pd.DataFrame(
            reconstruct(fit, yields.to_numpy(dtype=float)),
            index=yields.index,
            columns=yields.columns,
        )
        hidden_df = pd.DataFrame(
            encode(fit, yields.to_numpy(dtype=float)),
            index=yields.index,
            columns=[f"Node {i + 1}" for i in range(hidden)],
        )
        corr = correlation_table(hidden_df, proxies).abs()
        row = {
            **summarize_reconstruction_metrics(
                f"Autoencoder {hidden} hidden nodes", yields, reconstructed
            ),
            "hidden_nodes": hidden,
            "scale": args.scale,
            "epochs": robustness_epochs,
            "restarts": robustness_restarts,
            "best_seed": fit.seed,
            "level_max_abs_corr": float(corr["Level"].max()),
            "slope_max_abs_corr": float(corr["Slope 20Y-2Y"].max()),
            "curvature_max_abs_corr": float(corr["Curvature 2*10Y-2Y-20Y"].max()),
        }
        rows.append(row)

    if args.scale == "none":
        scaled_fit = train_autoencoder(
            yields.to_numpy(dtype=float),
            hidden=args.hidden,
            epochs=robustness_epochs,
            learning_rate=0.01,
            restarts=robustness_restarts,
            seed=args.seed + 30_000,
            scale="standardize",
            batch_size=args.batch_size,
        )
        scaled_reconstructed = pd.DataFrame(
            reconstruct(scaled_fit, yields.to_numpy(dtype=float)),
            index=yields.index,
            columns=yields.columns,
        )
        scaled_hidden = pd.DataFrame(
            encode(scaled_fit, yields.to_numpy(dtype=float)),
            index=yields.index,
            columns=[f"Node {i + 1}" for i in range(args.hidden)],
        )
        scaled_corr = correlation_table(scaled_hidden, proxies).abs()
        rows.append(
            {
                **summarize_reconstruction_metrics(
                    "Autoencoder 3 hidden nodes, standardized sensitivity",
                    yields,
                    scaled_reconstructed,
                ),
                "hidden_nodes": args.hidden,
                "scale": "standardize",
                "epochs": robustness_epochs,
                "restarts": robustness_restarts,
                "best_seed": scaled_fit.seed,
                "level_max_abs_corr": float(scaled_corr["Level"].max()),
                "slope_max_abs_corr": float(scaled_corr["Slope 20Y-2Y"].max()),
                "curvature_max_abs_corr": float(
                    scaled_corr["Curvature 2*10Y-2Y-20Y"].max()
                ),
            }
        )

    robustness = pd.DataFrame(rows)
    robustness.to_csv(output_dir / "robustness_summary.csv", index=False)
    return robustness


def trend_follow_positions(yields: pd.DataFrame) -> pd.DataFrame:
    # If yields fell last week, trend-following expects further decline and goes long.
    weekly_change = yields.diff()
    return pd.DataFrame(
        np.where(weekly_change < 0, 1.0, -1.0),
        index=yields.index,
        columns=yields.columns,
    ).iloc[1:]


def position_returns_bp(
    yields: pd.DataFrame,
    positions: pd.DataFrame,
    hold_weeks: int,
) -> pd.DataFrame:
    future_yields = yields.shift(-hold_weeks)
    aligned_yields = yields.reindex(positions.index)
    future_yields = future_yields.reindex(positions.index)
    returns = positions * (aligned_yields - future_yields) * 100.0
    return returns.dropna(how="any")


def make_forecast_training_arrays(
    values: np.ndarray,
    lags: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for current_pos in range(lags - 1, len(values) - horizon):
        x_rows.append(values[current_pos - lags + 1 : current_pos + 1])
        y_rows.append(values[current_pos + horizon])
    if not x_rows:
        n_features = values.shape[1]
        return (
            np.empty((0, lags, n_features), dtype=float),
            np.empty((0, n_features), dtype=float),
        )
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def make_forecast_prediction_arrays(
    values: np.ndarray,
    positions: np.ndarray,
    lags: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_positions = [
        pos for pos in positions if pos >= lags - 1 and pos + horizon < len(values)
    ]
    if not valid_positions:
        n_features = values.shape[1]
        return np.empty((0,), dtype=int), np.empty((0, lags, n_features), dtype=float)

    x_rows = [
        values[pos - lags + 1 : pos + 1]
        for pos in valid_positions
    ]
    return np.asarray(valid_positions, dtype=int), np.asarray(x_rows, dtype=float)


def forecast_strategy_returns_bp(
    yields: pd.DataFrame,
    forecast: pd.DataFrame,
    hold_weeks: int,
) -> pd.DataFrame:
    current = yields.reindex(forecast.index)
    future = yields.shift(-hold_weeks).reindex(forecast.index)
    positions = pd.DataFrame(
        np.where(forecast < current, 1.0, -1.0),
        index=forecast.index,
        columns=forecast.columns,
    )
    returns = positions * (current - future) * 100.0
    return returns.dropna(how="any")


def fit_predict_var(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        [x_train.reshape(len(x_train), -1), np.ones(len(x_train))]
    )
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    predict_design = np.column_stack(
        [x_predict.reshape(len(x_predict), -1), np.ones(len(x_predict))]
    )
    return predict_design @ coef


def fit_predict_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    units: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    tf = import_keras()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    mean = y_train.mean(axis=0, keepdims=True)
    std = y_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x_train_scaled = ((x_train - mean) / std).astype(np.float32)
    y_train_scaled = ((y_train - mean) / std).astype(np.float32)
    x_predict_scaled = ((x_predict - mean) / std).astype(np.float32)
    effective_batch_size = len(x_train_scaled) if batch_size <= 0 else min(batch_size, len(x_train_scaled))

    inputs = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2]), name="yield_history")
    hidden = tf.keras.layers.LSTM(units, name="lstm")(inputs)
    outputs = tf.keras.layers.Dense(x_train.shape[2], activation="linear", name="forecast")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="yield_forecast_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    history = model.fit(
        x_train_scaled,
        y_train_scaled,
        epochs=epochs,
        batch_size=effective_batch_size,
        shuffle=False,
        verbose=0,
    )
    forecast_scaled = model(x_predict_scaled, training=False).numpy()
    forecast = forecast_scaled * std + mean
    return forecast, float(history.history["loss"][-1])


def run_lstm_var_trading(
    yields: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    ae_returns: pd.DataFrame | None = None,
    trend_returns: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    values = yields.to_numpy(dtype=float)
    returns_by_method: Dict[str, List[pd.DataFrame]] = {"LSTM": [], "VAR": []}
    training_rows: List[Dict[str, float | int | str]] = []
    start_year = yields.index.min().year + 5
    end_year = yields.index.max().year

    for year in range(start_year, end_year + 1):
        invest_dates = yields.loc[str(year)].index
        if len(invest_dates) == 0:
            continue

        train_end = invest_dates[0]
        train_start = train_end - pd.DateOffset(years=5)
        train = yields.loc[(yields.index >= train_start) & (yields.index < train_end)]
        x_train, y_train = make_forecast_training_arrays(
            train.to_numpy(dtype=float),
            lags=args.forecast_lags,
            horizon=args.hold_weeks,
        )
        if len(x_train) < 40:
            continue

        invest_positions = yields.index.get_indexer(invest_dates)
        valid_positions, x_predict = make_forecast_prediction_arrays(
            values,
            invest_positions,
            lags=args.forecast_lags,
            horizon=args.hold_weeks,
        )
        if len(valid_positions) == 0:
            continue

        forecast_index = yields.index[valid_positions]
        var_forecast = pd.DataFrame(
            fit_predict_var(x_train, y_train, x_predict),
            index=forecast_index,
            columns=yields.columns,
        )
        returns_by_method["VAR"].append(
            forecast_strategy_returns_bp(yields, var_forecast, args.hold_weeks)
        )

        lstm_forecast_values, lstm_loss = fit_predict_lstm(
            x_train=x_train,
            y_train=y_train,
            x_predict=x_predict,
            units=args.lstm_units,
            epochs=args.lstm_epochs,
            learning_rate=args.lstm_learning_rate,
            batch_size=args.forecast_batch_size,
            seed=args.seed + 40_000 + year,
        )
        lstm_forecast = pd.DataFrame(
            lstm_forecast_values,
            index=forecast_index,
            columns=yields.columns,
        )
        returns_by_method["LSTM"].append(
            forecast_strategy_returns_bp(yields, lstm_forecast, args.hold_weeks)
        )
        training_rows.append(
            {
                "year": year,
                "train_start": str(train.index.min().date()),
                "train_end": str(train.index.max().date()),
                "train_samples": len(x_train),
                "investment_samples": len(forecast_index),
                "lstm_loss": lstm_loss,
                "lstm_epochs": args.lstm_epochs,
                "lstm_units": args.lstm_units,
                "forecast_lags": args.forecast_lags,
                "hold_weeks": args.hold_weeks,
            }
        )

    method_returns: Dict[str, pd.DataFrame] = {}
    for method, frames in returns_by_method.items():
        if frames:
            method_returns[method] = pd.concat(frames).sort_index()

    if ae_returns is not None and not ae_returns.empty:
        method_returns["Autoencoder"] = ae_returns
    if trend_returns is not None and not trend_returns.empty:
        method_returns["Trend-follow"] = trend_returns

    if not method_returns:
        empty = pd.DataFrame(columns=yields.columns)
        return empty, empty

    common_index = sorted(set.intersection(*(set(frame.index) for frame in method_returns.values())))
    method_returns = {
        method: frame.loc[common_index]
        for method, frame in method_returns.items()
    }
    combined = pd.concat(method_returns, axis=1)
    combined.to_csv(output_dir / "lstm_var_strategy_returns_bp.csv")

    average = pd.concat(
        {method: frame.mean() for method, frame in method_returns.items()},
        axis=1,
    )
    average.index.name = "maturity"
    average.to_csv(output_dir / "lstm_var_average_capital_gain.csv")

    pd.DataFrame(training_rows).to_csv(
        output_dir / "lstm_var_training_summary.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot_methods = [method for method in ["LSTM", "VAR", "Autoencoder"] if method in method_returns]
    for maturity in ["10Y", "20Y"]:
        for method in plot_methods:
            cumulative = method_returns[method][maturity].cumsum()
            ax.plot(cumulative.index, cumulative, label=f"{method} {maturity}", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative Returns: LSTM, VAR, and Autoencoder")
    ax.set_ylabel("Cumulative gain (bp)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "lstm_var_cumulative_returns.png", dpi=180)
    plt.close(fig)

    return average, combined


def rolling_autoencoder_strategy(
    yields: pd.DataFrame,
    hidden: int,
    learning_years: int,
    epochs: int,
    learning_rate: float,
    restarts: int,
    seed: int,
    scale: str,
    batch_size: int,
    hold_weeks: int,
) -> pd.DataFrame:
    returns_by_year = []
    start_year = yields.index.min().year + learning_years
    end_year = yields.index.max().year

    for year in range(start_year, end_year + 1):
        invest_dates = yields.loc[str(year)].index
        if len(invest_dates) == 0:
            continue

        train_end = invest_dates[0]
        train_start = train_end - pd.DateOffset(years=learning_years)
        train = yields.loc[(yields.index >= train_start) & (yields.index < train_end)]
        if len(train) < 80:
            continue

        fit = train_autoencoder(
            train.to_numpy(dtype=float),
            hidden=hidden,
            epochs=epochs,
            learning_rate=learning_rate,
            restarts=restarts,
            seed=seed + year,
            scale=scale,
            batch_size=batch_size,
        )
        x_invest = yields.loc[invest_dates].to_numpy(dtype=float)
        x_hat = reconstruct(fit, x_invest)
        reconstructed = pd.DataFrame(x_hat, index=invest_dates, columns=yields.columns)
        actual = yields.loc[invest_dates]
        positions = pd.DataFrame(
            np.where(actual > reconstructed, 1.0, -1.0),
            index=invest_dates,
            columns=yields.columns,
        )
        returns = position_returns_bp(yields, positions, hold_weeks=hold_weeks)
        returns_by_year.append(returns)

    if not returns_by_year:
        return pd.DataFrame(columns=yields.columns)
    return pd.concat(returns_by_year).sort_index()


def run_trading(
    yields: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ae_returns = rolling_autoencoder_strategy(
        yields=yields,
        hidden=args.hidden,
        learning_years=5,
        epochs=args.trading_epochs,
        learning_rate=args.learning_rate,
        restarts=max(2, args.restarts // 2),
        seed=args.seed,
        scale=args.scale,
        batch_size=args.batch_size,
        hold_weeks=args.hold_weeks,
    )
    trend_positions = trend_follow_positions(yields)
    trend_returns = position_returns_bp(yields, trend_positions, hold_weeks=args.hold_weeks)

    common_index = ae_returns.index.intersection(trend_returns.index)
    ae_returns = ae_returns.loc[common_index]
    trend_returns = trend_returns.loc[common_index]

    average = pd.concat(
        {
            "Autoencoder": ae_returns.mean(),
            "Trend-follow": trend_returns.mean(),
            "Always long": position_returns_bp(
                yields,
                pd.DataFrame(1.0, index=common_index, columns=yields.columns),
                args.hold_weeks,
            ).mean(),
            "Always short": position_returns_bp(
                yields,
                pd.DataFrame(-1.0, index=common_index, columns=yields.columns),
                args.hold_weeks,
            ).mean(),
        },
        axis=1,
    )
    average.index.name = "maturity"
    average.to_csv(output_dir / "trading_average_capital_gain.csv")

    combined = pd.concat(
        {
            "autoencoder": ae_returns,
            "trend_follow": trend_returns,
        },
        axis=1,
    )
    combined.to_csv(output_dir / "strategy_returns_bp.csv")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for maturity in ["10Y", "20Y"]:
        ae_cum = ae_returns[maturity].cumsum()
        trend_cum = trend_returns[maturity].cumsum()
        ax.plot(ae_cum.index, ae_cum, label=f"AE {maturity}", linewidth=1.2)
        ax.plot(
            trend_cum.index,
            trend_cum,
            label=f"Trend {maturity}",
            linewidth=1.0,
            linestyle="--",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative One-Month Capital Gains")
    ax.set_ylabel("Cumulative gain (bp)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "trading_cumulative_returns.png", dpi=180)
    plt.close(fig)

    return average, combined


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    yields = load_jgb_yields(args.csv, args.start_date, args.end_date)
    yields.to_csv(args.output_dir / "weekly_jgb_yields.csv")
    plot_yield_history(yields, args.output_dir / "yield_history.png")

    pca_table, loadings, _ = run_pca(yields)
    pca_table.to_csv(args.output_dir / "pca_explained_variance.csv", index=False)
    loadings.to_csv(args.output_dir / "pca_loadings.csv")
    plot_pca(pca_table, loadings, args.output_dir)
    pca_three_reconstructed = pca_reconstruction(yields, components=3)
    pca_metrics = reconstruction_metrics(yields, pca_three_reconstructed)
    pca_metrics.to_csv(args.output_dir / "pca_reconstruction_metrics.csv")

    fit = train_autoencoder(
        yields.to_numpy(dtype=float),
        hidden=args.hidden,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        restarts=args.restarts,
        seed=args.seed,
        scale=args.scale,
        batch_size=args.batch_size,
    )
    reconstructed_values = reconstruct(fit, yields.to_numpy(dtype=float))
    reconstructed = pd.DataFrame(reconstructed_values, index=yields.index, columns=yields.columns)
    reconstructed.to_csv(args.output_dir / "autoencoder_reconstructed_yields.csv")

    metrics = reconstruction_metrics(yields, reconstructed)
    metrics.to_csv(args.output_dir / "reconstruction_metrics.csv")

    proxies = make_proxies(yields)
    hidden = encode(fit, yields.to_numpy(dtype=float))
    hidden_aligned, decoder_aligned, node_corr = align_hidden_factors(
        hidden, proxies, fit.w_decoder
    )
    factor_corr = correlation_table(hidden_aligned, proxies)
    hidden_aligned.to_csv(args.output_dir / "autoencoder_hidden_factors.csv")
    decoder_aligned.to_csv(args.output_dir / "autoencoder_decoder_loadings.csv")
    node_corr.to_csv(args.output_dir / "node_proxy_correlations.csv")
    factor_corr.to_csv(args.output_dir / "factor_proxy_correlations.csv")

    plot_autoencoder_results(
        yields=yields,
        reconstructed=reconstructed,
        decoder=decoder_aligned,
        hidden_aligned=hidden_aligned,
        proxies=proxies,
        output_dir=args.output_dir,
    )

    validation = run_temporal_validation(yields, args, args.output_dir)
    robustness = None
    if not args.skip_robustness:
        robustness = run_robustness_checks(
            yields=yields,
            pca_three_reconstructed=pca_three_reconstructed,
            args=args,
            output_dir=args.output_dir,
        )

    trading_average = None
    trading_returns = None
    if not args.skip_trading:
        trading_average, trading_returns = run_trading(yields, args, args.output_dir)

    forecast_average = None
    forecast_returns = None
    if args.run_forecast_models:
        ae_returns = None
        trend_returns = None
        if trading_returns is not None and not trading_returns.empty:
            ae_returns = trading_returns["autoencoder"]
            trend_returns = trading_returns["trend_follow"]
        forecast_average, forecast_returns = run_lstm_var_trading(
            yields=yields,
            args=args,
            output_dir=args.output_dir,
            ae_returns=ae_returns,
            trend_returns=trend_returns,
        )

    config = {
        "sample_start": args.start_date,
        "sample_end": args.end_date,
        "maturities": MATURITIES,
        "weekly_observations": int(len(yields)),
        "hidden_nodes": args.hidden,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "scale": args.scale,
        "restarts": args.restarts,
        "seed": args.seed,
        "final_training_loss": fit.loss_history[-1],
        "best_seed": fit.seed,
        "temporal_validation_holdout_rmse_bp": float(
            validation.loc[
                validation["model"] == "Autoencoder temporal holdout", "mean_rmse_bp"
            ].iloc[0]
        ),
        "robustness_skipped": bool(args.skip_robustness),
        "trading_skipped": bool(args.skip_trading),
        "forecast_models_run": bool(args.run_forecast_models),
    }
    if trading_returns is not None:
        config["trading_observations"] = int(len(trading_returns))
    if forecast_returns is not None:
        config["forecast_model_observations"] = int(len(forecast_returns))
        config["forecast_lags"] = args.forecast_lags
        config["lstm_units"] = args.lstm_units
        config["lstm_epochs"] = args.lstm_epochs
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("Replication complete.")
    print(f"Weekly observations: {len(yields):,}")
    print(
        "PCA cumulative variance with 3 components: "
        f"{pca_table.loc[pca_table['component'] == 3, 'cumulative_variance'].iloc[0]:.4f}"
    )
    print(f"Mean reconstruction RMSE: {metrics['rmse_bp'].mean():.2f} bp")
    print(
        "Temporal holdout RMSE: "
        f"{validation.loc[validation['model'] == 'Autoencoder temporal holdout', 'mean_rmse_bp'].iloc[0]:.2f} bp"
    )
    if forecast_average is not None:
        print("LSTM/VAR trading comparison complete.")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
