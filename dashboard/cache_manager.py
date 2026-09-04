import hashlib
import json
import os
import pickle
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / '.cache'
PRED_DIR = CACHE_DIR / 'predictions'
PORT_DIR = CACHE_DIR / 'portfolios'


def _ensure_dirs():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PORT_DIR.mkdir(parents=True, exist_ok=True)


def _make_key(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """A stable id for the dataset a backtest runs on.

    Predictions are only valid for the exact panel they were computed on, so
    the fingerprint must change whenever the universe (permno set) or the data
    (row count, month range) changes. Without this the cache would silently
    serve predictions from a previous dataset after a rebuild or universe
    switch — the permnos would no longer match the loaded panel.
    """
    raw = json.dumps({
        'n_rows': int(len(df)),
        'permnos': sorted(int(p) for p in df['permno'].unique()),
        'months': sorted(str(m) for m in df['ym'].unique()),
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def prediction_key(model_type, model_params, retrain_every, feature_cols=None, window_type="expanding", auto_tune=False, data_fingerprint=None, oos_start=None, rolling_window=None):
    return _make_key({
        # Bump when the shape of a stored prediction frame changes. v2 adds
        # ivol_xs, which portfolio construction needs for the volatility cap;
        # frames cached before it lack the column.
        'predictions_version': 2,
        'model_type': model_type,
        'model_params': model_params,
        'retrain_every': retrain_every,
        'feature_cols': sorted(feature_cols) if feature_cols else None,
        'window_type': window_type,
        'auto_tune': auto_tune,
        'data_fingerprint': data_fingerprint,
        # oos_start and rolling_window both change the walk-forward predictions
        # (which months are predicted, and the retrain/training windows), so
        # they must be part of the key or a new start silently reuses old preds.
        'oos_start': oos_start,
        'rolling_window': rolling_window,
    })


def portfolio_key(pred_key, K, vol_tilt, regime_lookback, strategy_type="long_only", K_short=10, construction_method="equal_weight", tc_bps=0.0, cost_bps=10.0, max_ivol_xs=None):
    return _make_key({
        # Bump when weight construction changes in a way the other key fields
        # cannot express. v2: ERC/MVO now receive a real point-in-time
        # covariance (previously always the identity) and the ERC objective is
        # scaled so it actually optimizes. Entries cached before that are wrong.
        'construction_version': 2,
        'pred_key': pred_key,
        'K': K,
        'vol_tilt': vol_tilt,
        'regime_lookback': regime_lookback,
        'strategy_type': strategy_type,
        'K_short': K_short,
        'construction_method': construction_method,
        'tc_bps': tc_bps,
        'cost_bps': cost_bps,
        'max_ivol_xs': max_ivol_xs,
    })


def get_predictions(key):
    _ensure_dirs()
    path = PRED_DIR / f'{key}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_predictions(key, predictions):
    _ensure_dirs()
    path = PRED_DIR / f'{key}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(predictions, f)


def get_portfolio(key):
    _ensure_dirs()
    path = PORT_DIR / f'{key}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_portfolio(key, portfolio):
    _ensure_dirs()
    path = PORT_DIR / f'{key}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(portfolio, f)
