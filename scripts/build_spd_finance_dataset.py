#!/usr/bin/env python3
"""Download/cache Yahoo daily prices and build an audited SPD finance dataset.

Requires only numpy and pandas; never imports or starts the training pipeline.
Run with --download once, then omit it to reproduce from the cached snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    ("1306.T", "TOPIX ETF", "Japan equity", "Adj Close"),
    ("2510.T", "NOMURA-BPI domestic bond ETF (not pure JGB)", "Japan broad bonds", "Adj Close"),
    ("JPY=X", "USD/JPY", "FX (JPY per USD)", "Close"),
    ("EURJPY=X", "EUR/JPY", "FX (JPY per EUR)", "Close"),
    ("1540.T", "Physical gold trust (JPY)", "Gold", "Adj Close"),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(raw_dir, start, end, include_jgb_candidate=False):
    """One bounded request per symbol, no automatic retry or hidden repair."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    symbols = [a[0] for a in ASSETS]
    if include_jgb_candidate:
        symbols.append("2561.T")
    request = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
               "period2": int(pd.Timestamp(end, tz="UTC").timestamp()),
               "interval": "1d", "events": "div,splits"}
    manifest = {"source": "Yahoo Finance chart API", "start": start,
                "end_exclusive": end, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": {}}
    for symbol in symbols:
        path = raw_dir / f"{symbol}.json"
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{urlencode(request)}"
        result = subprocess.run(
            ["curl", "--silent", "--show-error", "--fail", "--location",
             "--max-time", "30", "--user-agent", "Mozilla/5.0", url],
            capture_output=True, check=True, timeout=35,
        )
        payload = json.loads(result.stdout)
        if payload["chart"].get("error") or not payload["chart"].get("result"):
            raise ValueError(f"No chart data for {symbol}")
        path.write_bytes(result.stdout)
        manifest["files"][symbol] = {"url": url, "sha256": digest(path)}
        print(f"Downloaded {symbol}: {len(result.stdout)} bytes", flush=True)
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_prices(path, start, end, price_field):
    chart = json.loads(path.read_text())["chart"]["result"][0]
    meta = chart["meta"]
    # Yahoo daily timestamps are exchange-local session labels, not UTC dates.
    index = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
    index = index.tz_convert(meta["exchangeTimezoneName"]).tz_localize(None).normalize()
    indicators = chart["indicators"]
    close = indicators["quote"][0]["close"]
    adjusted = indicators.get("adjclose", [{}])[0].get("adjclose")
    if price_field == "Adj Close" and adjusted is None:
        raise ValueError(f"Required adjusted price absent: {path.name}")
    frame = pd.DataFrame({"Close": close, "Adj Close": adjusted if adjusted is not None else np.nan}, index=index)
    frame = frame.sort_index().loc[lambda x: (x.index >= start) & (x.index < end)]
    # Yahoo can append a duplicate live bar beyond period2. Exclude it before
    # duplicate validation; never silently choose among historical duplicates.
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate session dates: {path.name}")
    series = frame[price_field].astype(float)
    valid = series.where(np.isfinite(series) & (series > 0))
    if valid.dropna().empty:
        raise ValueError(f"No usable prices: {path.name}")
    valid = valid.loc[valid.first_valid_index():valid.last_valid_index()]
    weekdays = pd.bdate_range(valid.index[0], valid.index[-1])
    audit = {
        "first_valid_date": str(valid.dropna().index[0].date()),
        "last_valid_date": str(valid.dropna().index[-1].date()),
        "valid_observations": int(valid.notna().sum()),
        "provider_rows": len(frame), "invalid_provider_rows": int(series.isna().sum() + ((series <= 0) | np.isinf(series)).sum()),
        "weekday_missing_in_active_range": int(valid.reindex(weekdays).isna().sum()),
        "weekday_missing_note": "Includes exchange holidays; not an exchange-calendar error count.",
        "adjusted_price_available": adjusted is not None,
        "adjusted_valid_observations": int((np.isfinite(frame['Adj Close']) & (frame['Adj Close'] > 0)).sum()),
        "price_field": price_field, "currency": meta.get("currency"),
        "exchange_timezone": meta["exchangeTimezoneName"],
        "provider_first_trade_date": str(pd.Timestamp(meta["firstTradeDate"], unit="s", tz="UTC").date()),
        "sha256": digest(path),
    }
    return valid, audit


def rolling_covariances(prices, window):
    """60 consecutive reference sessions, ddof=1, no annualization or filling."""
    if window < 2:
        raise ValueError("window must be >= 2")
    returns = np.log(prices).diff().iloc[1:]
    values = returns.to_numpy(dtype=np.float64)
    matrices, endpoints = [], []
    for end in range(window - 1, len(values)):
        block = values[end - window + 1:end + 1]
        if not np.isfinite(block).all():
            continue
        centered = block - block.mean(axis=0)
        cov = centered.T @ centered / (window - 1)
        matrices.append((cov + cov.T) / 2)
        endpoints.append(end)
    if not matrices:
        raise ValueError("No complete rolling windows")
    return np.stack(matrices), returns, np.asarray(endpoints, dtype=np.int64)


def quarantine_prices(prices, rules):
    """Apply reviewed, dated rules, retaining reference calendar rows."""
    prices = prices.copy()
    applied = []
    for rule in rules.get("quarantine", []):
        low, high = rule["adjusted_price_range"]
        for date in rule["dates"]:
            date = pd.Timestamp(date)
            if date in prices.index:
                value = prices.loc[date, rule["ticker"]]
                if low <= value <= high:
                    applied.append({"ticker": rule["ticker"], "date": str(date.date()),
                                    "original_adjusted_price": float(value), "reason": rule["reason"]})
                    prices.loc[date, rule["ticker"]] = np.nan
    changes = np.log(prices).diff()
    if (changes.abs() > rules["max_abs_log_return"]).any().any():
        raise ValueError("Unreviewed extreme price change: inspect raw prices/corporate actions; do not blindly regularize or clip")
    return prices, applied


def validate_spd(covariances, relative_eigen_floor=1e-12, allow_regularization=False):
    """Fail by default; optional per-matrix minimal eps relative to its scale."""
    if not np.isfinite(relative_eigen_floor) or not 0 < relative_eigen_floor < 1:
        raise ValueError("relative_eigen_floor must be in (0, 1)")
    cov = np.asarray(covariances, dtype=np.float64).copy()
    if cov.ndim != 3 or cov.shape[-1] != cov.shape[-2] or not np.isfinite(cov).all():
        raise ValueError("Expected finite (N,d,d) covariances")
    symmetry = np.max(np.abs(cov - cov.swapaxes(-1, -2)), axis=(-2, -1))
    scale = np.max(np.abs(cov), axis=(-2, -1))
    if np.any(symmetry > 1e-12 * scale):
        raise ValueError("Non-symmetric covariances")
    eig_before = np.linalg.eigvalsh(cov)
    if np.any(eig_before[:, -1] <= 0) or np.any(eig_before[:, 0] < -1e-12 * eig_before[:, -1]):
        raise ValueError("Covariance is materially indefinite or has zero variance")
    floor = relative_eigen_floor * eig_before[:, -1]
    eps = np.maximum(0.0, floor - eig_before[:, 0])
    if np.any(eps > 0) and not allow_regularization:
        raise ValueError(f"{np.count_nonzero(eps)} windows need regularization; inspect data, then explicitly use --allow-regularization")
    cov += eps[:, None, None] * np.eye(cov.shape[-1])
    eig = np.linalg.eigvalsh(cov)
    sign, logdet = np.linalg.slogdet(cov)
    determinant = np.linalg.det(cov)
    condition = eig[:, -1] / eig[:, 0]
    if np.any(eig[:, 0] <= 0) or np.any(sign != 1) or not np.isfinite(logdet).all() or not np.isfinite(condition).all():
        raise ValueError("SPD validation failed after regularization")
    report = {"symmetry_max_abs": float(symmetry.max()),
              "minimum_eigenvalue_before": float(eig_before.min()),
              "minimum_eigenvalue": float(eig.min()), "maximum_eigenvalue": float(eig.max()),
              "determinant_min": float(determinant.min()), "determinant_max": float(determinant.max()),
              "logdet_min": float(logdet.min()), "logdet_max": float(logdet.max()),
              "condition_number_max": float(condition.max()), "condition_number_median": float(np.median(condition)),
              "all_spd": True, "relative_eigen_floor": relative_eigen_floor,
              "regularized_count": int(np.count_nonzero(eps)), "eps_max": float(eps.max())}
    return cov, eps, eig, determinant, condition, report


def chronological_indices(endpoints, window, fractions=(0.7, 0.15, 0.15)):
    """Split by time, then purge shared return AND boundary price observations."""
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Three positive split fractions must sum to 1")
    endpoints = np.asarray(endpoints)
    if np.any(np.diff(endpoints) <= 0):
        raise ValueError("Endpoints must be strictly increasing")
    n = len(endpoints)
    cuts = [0, int(n * fractions[0]), int(n * fractions[0]) + int(n * fractions[1]), n]
    indices = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        idx = np.arange(left, right, dtype=np.int64)
        if indices:
            # A window ends at return e and uses price indices [e-w+1,e+1].
            idx = idx[endpoints[idx] - window + 1 > endpoints[indices[-1][-1]] + 1]
        if not len(idx):
            raise ValueError("Empty split after purging; increase history or adjust fractions")
        indices.append(idx)
    return dict(zip(("train", "val", "test"), indices))


def build(args):
    raw_dir = args.output_dir / "raw"
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    if args.start < manifest["start"] or args.end > manifest["end_exclusive"]:
        raise ValueError("Cached download does not cover requested range; use --download")
    quality_rules = json.loads(args.quality_rules.read_text())
    series, audits, observation_dates = {}, {}, {}
    for ticker, name, asset_class, field in ASSETS:
        path = raw_dir / f"{ticker}.json"
        if digest(path) != manifest["files"][ticker]["sha256"]:
            raise ValueError(f"Raw snapshot hash mismatch: {ticker}")
        s, audit = read_prices(path, args.start, args.end, field)
        audit.update(ticker=ticker, asset_name=name, asset_class=asset_class)
        if ticker in quality_rules.get("not_before", {}):
            first_date = quality_rules["not_before"][ticker]["date"]
            audit["pre_listing_observations_excluded"] = int(s.loc[s.index < first_date].notna().sum())
            s = s.loc[first_date:]
        observed = pd.Series(s.index, index=s.index).where(s.notna())
        if ticker.endswith("=X"):
            # Lag by one provider session, preserving missing rows (no ffill).
            s, observed = s.shift(1), observed.shift(1)
        series[ticker], observation_dates[ticker], audits[ticker] = s, observed, audit
    # Reference trading sessions: actual valid TOPIX ETF observations.
    start = max(s.first_valid_index() for s in series.values())
    end = min(s.last_valid_index() for s in series.values())
    calendar = series["1306.T"].dropna().loc[start:end].index
    prices = pd.DataFrame({k: s.reindex(calendar) for k, s in series.items()}, index=calendar)
    observed = pd.DataFrame({k: s.reindex(calendar) for k, s in observation_dates.items()}, index=calendar)
    for ticker in series:
        audits[ticker]["missing_on_common_topix_sessions"] = int(prices[ticker].isna().sum())
        audits[ticker]["zero_price_changes_on_common_sessions"] = int(prices[ticker].diff().eq(0).sum())
    prices, quarantined = quarantine_prices(prices, quality_rules)
    cov, returns, endpoints = rolling_covariances(prices, args.window)
    cov, eps, eig, det, cond, validation = validate_spd(cov, args.relative_eigen_floor, args.allow_regularization)
    splits = chronological_indices(endpoints, args.window, args.splits)
    dates = returns.index.to_numpy(dtype="datetime64[D]")[endpoints]
    split_report = {name: {"count": len(idx), "first_date": str(dates[idx[0]]), "last_date": str(dates[idx[-1]])} for name, idx in splits.items()}
    metadata = {
        "schema_version": 1, "source": manifest, "assets": audits,
        "requested_start": args.start, "end_exclusive": args.end,
        "price_date_range": [str(calendar[0].date()), str(calendar[-1].date())],
        "shape": list(cov.shape), "dimension": 5, "intrinsic_dimension": 15,
        "rolling_window": args.window, "frequency": "daily TOPIX-observed sessions",
        "covariance": "log price differences, sample covariance ddof=1, float64, not annualized",
        "price_policy": "ETF adjusted close required; FX close lagged one provider session; no interpolation, forward/back fill, or price repair",
        "quality_rules": quality_rules, "quality_rules_sha256": digest(args.quality_rules),
        "quarantined_prices": quarantined,
        "maximum_abs_log_return_by_asset": {ticker: float(returns[ticker].abs().max()) for ticker in series},
        "missing_policy": "Returns computed BEFORE removing any session; discard windows with any nonfinite return",
        "fx_lag_provider_sessions": 1, "common_price_sessions": len(prices),
        "potential_windows": max(0, len(returns) - args.window + 1),
        "discarded_missing_windows": max(0, len(returns) - args.window + 1) - len(cov),
        "metric": "affine_invariant", "dataset_seed": args.dataset_seed,
        "seed_note": "Deterministic preprocessing/splitting; seed recorded but unused. Training seed is independent.",
        "split_fractions_before_purge": list(args.splits), "split_policy": "chronological, purge shared underlying prices and returns",
        "splits": split_report, "purged_samples": len(cov) - sum(len(idx) for idx in splits.values()),
        "spd_validation": validation,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "script_sha256": digest(__file__),
    }
    candidate = raw_dir / "2561.T.json"
    if "2561.T" in manifest["files"]:
        if digest(candidate) != manifest["files"]["2561.T"]["sha256"]:
            raise ValueError("Candidate raw snapshot hash mismatch")
        _, metadata["pure_jgb_candidate"] = read_prices(candidate, args.start, args.end, "Adj Close")
    output = args.output_dir / f"spd_finance_5asset_{args.window}d.npz"
    np.savez_compressed(
        output, covariances=cov, dates=dates, returns=returns.to_numpy(dtype=np.float64),
        return_dates=returns.index.to_numpy(dtype="datetime64[D]"),
        prices=prices.to_numpy(dtype=np.float64), price_dates=calendar.to_numpy(dtype="datetime64[D]"),
        price_observation_dates=observed.to_numpy(dtype="datetime64[D]"),
        window_start_return_index=endpoints - args.window + 1, window_end_return_index=endpoints,
        tickers=np.asarray([a[0] for a in ASSETS]), asset_names=np.asarray([a[1] for a in ASSETS]),
        regularization_eps=eps, eigenvalues=eig, determinants=det, condition_numbers=cond,
        **{f"{name}_indices": idx for name, idx in splits.items()}, metadata_json=np.asarray(json.dumps(metadata)),
    )
    metadata["npz_sha256"] = digest(output)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "shape": list(cov.shape), "splits": split_report,
                      "spd": validation}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-jgb-candidate", action="store_true")
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat(), help="Exclusive date; excludes today's incomplete bars")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "spd_finance")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--quality-rules", type=Path, default=ROOT / "config" / "dataset" / "spd_finance_quality.json")
    parser.add_argument("--splits", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--allow-regularization", action="store_true")
    parser.add_argument("--relative-eigen-floor", type=float, default=1e-12)
    args = parser.parse_args()
    args.start, args.end = pd.Timestamp(args.start).date().isoformat(), pd.Timestamp(args.end).date().isoformat()
    if args.start >= args.end:
        parser.error("start must precede end")
    if args.download:
        download(args.output_dir / "raw", args.start, args.end, args.include_jgb_candidate)
    build(args)


if __name__ == "__main__":
    main()
