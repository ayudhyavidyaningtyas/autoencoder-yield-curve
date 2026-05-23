# Autoencoder Yield Curve Replication

Replication of Suimon, Sakaji, Izumi, and Matsushima (2020),
"Autoencoder-Based Three-Factor Model for the Yield Curve of Japanese
Government Bonds and a Trading Strategy".

The project implements the paper's main empirical workflow:

- PCA benchmark for the 2Y, 5Y, 7Y, 10Y, 15Y, and 20Y yield curve.
- A shallow 6-3-6 autoencoder with tanh encoder and linear decoder.
- Interpretation of hidden factors as level, slope, and curvature.
- A compact long-short trading replication using reconstructed yields as fair
  value estimates.
- Robustness checks comparing preprocessing, hidden-node counts, and linear
  versus nonlinear bottlenecks.

## Data

The downloaded source file is:

```text
data/jgbcme_all.csv
```

Source: Japan Ministry of Finance historical JGB interest-rate data:

```text
https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv
```

The script filters to the paper's sample window by default:

```text
1992-07-01 to 2019-07-31
```

Daily observations are converted to weekly Friday observations using the last
available daily value in each week.

## Supplied US Dataset

The supplied US yield-curve file can also be used with the same methodology.
The alternate runner uses:

```text
data/USdataYC.csv
```

The supplied file has daily US maturities from 1M to 30Y. Because it does not
include 15Y, the alternate runner uses this six-maturity subset:

```text
2Y, 3Y, 5Y, 7Y, 10Y, 20Y
```

This preserves a six-node input layer and keeps the same level, slope
`20Y-2Y`, and curvature `2*10Y-2Y-20Y` proxy definitions.

## Setup

The autoencoder is implemented in TensorFlow/Keras to match the paper's software
choice. Use a TensorFlow-supported Python version, such as Python 3.11 or 3.12:

```bash
pip install -r requirements.txt
```

On this machine, the system `python3` is newer than TensorFlow's usual wheel
support. The verified local environment is:

```bash
.venv_tf/bin/python src/replicate_autoencoder_yield_curve.py
```

## Run

From this folder:

```bash
python3 src/replicate_autoencoder_yield_curve.py
```

For a faster check:

```bash
python3 src/replicate_autoencoder_yield_curve.py --epochs 800 --skip-trading
```

To reproduce the paper's LSTM and VAR trading comparison, add
`--run-forecast-models`. This is slower because it trains one LSTM each year in
the rolling backtest:

```bash
.venv_tf/bin/python src/replicate_autoencoder_yield_curve.py \
  --run-forecast-models \
  --lstm-epochs 80
```

To run the same methodology on the supplied US dataset:

```bash
.venv_tf/bin/python src/replicate_autoencoder_us_yield_curve.py
```

The US outputs are written to:

```text
outputs_us/
```

For the longer US LSTM/VAR comparison:

```bash
.venv_tf/bin/python src/replicate_autoencoder_us_yield_curve.py \
  --run-forecast-models
```

You can override the US CSV path if needed:

```bash
.venv_tf/bin/python src/replicate_autoencoder_us_yield_curve.py \
  --csv /path/to/USdataYC.csv
```

The default autoencoder uses centered yields, Keras bias terms, a tanh encoder,
a linear decoder, and PCA-based initialization. This keeps the model close to
the paper's architecture while making the reconstruction comparison with
mean-centered PCA fair. To run the raw no-bias sensitivity check, use raw
yields, no bias terms, and random initialization:

```bash
python3 src/replicate_autoencoder_yield_curve.py \
  --scale none \
  --no-bias \
  --random-init
```

For additional numerical checks, use `--scale standardize` or
`--encoder-activation linear`. The linear bottleneck check should nearly
match PCA because a linear 6-3-6 autoencoder and 3-component PCA span the same
kind of rank-3 reconstruction problem.

## Outputs

The script writes to `outputs/`:

- `yield_history.png`: weekly yield history by maturity.
- `pca_explained_variance.png`: cumulative PCA variance.
- `pca_loadings.png`: PCA loading patterns by maturity.
- `autoencoder_decoder_loadings.png`: decoder weights used to interpret hidden nodes.
- `autoencoder_factor_proxies.png`: hidden factors compared with yield-curve proxies.
- `autoencoder_orthogonalized_decoder_loadings.png`: rotated decoder weights after orthogonalizing the hidden factor space.
- `autoencoder_orthogonalized_factor_proxies.png`: orthogonalized factors compared with level, slope, and curvature proxies.
- `reconstruction_fit.png`: actual vs reconstructed yields for selected maturities.
- `trading_cumulative_returns.png`: cumulative one-month capital gains.
- `model_comparison_summary.csv`: PCA vs autoencoder reconstruction summary.
- `pca_explained_variance.csv`: PCA variance table.
- `pca_reconstruction_metrics.csv`: three-component PCA reconstruction error.
- `reconstruction_metrics.csv`: autoencoder RMSE/MAE by maturity.
- `temporal_validation_metrics.csv`: 80/20 chronological train/holdout reconstruction check.
- `robustness_summary.csv`: PCA, hidden-node-count, and scaling sensitivity checks.
- `autoencoder_orthogonalization_summary.csv`: cross-correlation before and after orthogonalizing hidden factors.
- `autoencoder_factor_cross_correlations.csv`: cross-correlations among aligned hidden factors.
- `factor_proxy_correlations.csv`: correlations between hidden factors and financial proxies.
- `node_proxy_correlations.csv`: raw hidden-node correlations before relabelling.
- `trading_average_capital_gain.csv`: average capital gain by maturity and strategy.
- `lstm_var_average_capital_gain.csv`: LSTM, VAR, and optional autoencoder average capital gains.
- `lstm_var_strategy_returns_bp.csv`: weekly strategy return series for LSTM/VAR comparison.
- `lstm_var_training_summary.csv`: annual LSTM/VAR training-window audit table.
- `lstm_var_cumulative_returns.png`: cumulative 10Y/20Y LSTM, VAR, and autoencoder comparison.
