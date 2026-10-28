"""
src/inject_experiment.py
Graded Botnet Injection Experiment — Neris family, 0% to 100%.

For each injection rate (10, 20, 100 %):
  1. Sample (injection_rate %) Neris flows + remaining benign flows (total = TOTAL_FLOWS)
  2. Run two-phase cascade inference on every flow
  3. Compute TPR, FPR, latency, and cascade category breakdown
  4. Save per-run CSVs + aggregated results + charts

Usage:
    python src/inject_experiment.py
    python src/inject_experiment.py --total-flows 5000 --seed 42
    python src/inject_experiment.py --binary-model lightgbm --anomaly-model isolation_forest
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROJECT_ROOT, PROCESSED_DIR, RANDOM_SEED
from logging_utils import get_logger
from cascade_inference import CascadeEngine

logger = get_logger("inject_experiment")

INJECTION_RATES  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
INJECTION_FAMILY = "Neris"
TOTAL_FLOWS      = 10_000
CHUNKSIZE        = 500_000

OUT_DIR   = PROJECT_ROOT / "reports" / "injection"
POOL_DIR  = PROJECT_ROOT / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)
POOL_DIR.mkdir(parents=True, exist_ok=True)

NERIS_CACHE  = POOL_DIR / "pool_neris.csv"
BENIGN_CACHE = POOL_DIR / "pool_benign_sample.csv"
BENIGN_CACHE_MAX = 200_000  # cap benign cache size to keep it small


def _load_family_pools(seed: int, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Neris and Benign pools. Caches to disk for fast re-runs."""
    if use_cache and NERIS_CACHE.exists() and BENIGN_CACHE.exists():
        logger.info("Loading pools from cache (fast path) ...")
        neris_df  = pd.read_csv(NERIS_CACHE,  low_memory=False)
        benign_df = pd.read_csv(BENIGN_CACHE, low_memory=False)
        logger.info("Cache hit — Neris: %d  Benign: %d", len(neris_df), len(benign_df))
        return neris_df, benign_df

    logger.info("Streaming processed CSV to build pools (one-time, ~4 min) ...")
    neris_chunks: list[pd.DataFrame] = []
    benign_chunks: list[pd.DataFrame] = []
    rng = __import__('numpy').random.default_rng(seed)
    for chunk in pd.read_csv(
        PROCESSED_DIR / "ctu13_flows.csv", chunksize=CHUNKSIZE, low_memory=False,
        dtype={"binary_label": "int8", "scenario_id": "int16",
               "total_packets": "float32", "total_bytes": "float32",
               "src_bytes": "float32", "dst_bytes": "float32",
               "duration": "float32", "src_port": "float32", "dst_port": "float32"},
    ):
        neris_chunks.append(chunk[chunk["family"] == INJECTION_FAMILY])
        # Subsample benign to avoid storing 19M rows
        ben_chunk = chunk[chunk["family"] == "Benign"]
        if len(ben_chunk) > 5000:
            ben_chunk = ben_chunk.sample(n=5000, random_state=int(rng.integers(1e6)))
        benign_chunks.append(ben_chunk)

    neris_df  = pd.concat(neris_chunks,  ignore_index=True)
    benign_df = pd.concat(benign_chunks, ignore_index=True)
    if len(benign_df) > BENIGN_CACHE_MAX:
        benign_df = benign_df.sample(n=BENIGN_CACHE_MAX, random_state=seed)

    # Cache to disk
    neris_df.to_csv(NERIS_CACHE,  index=False)
    benign_df.to_csv(BENIGN_CACHE, index=False)
    logger.info("Pools cached — Neris: %d  Benign: %d", len(neris_df), len(benign_df))
    logger.info("  Neris cache  -> %s", NERIS_CACHE)
    logger.info("  Benign cache -> %s", BENIGN_CACHE)
    return neris_df, benign_df


def _build_mix(neris_df, benign_df, injection_pct, total, seed):
    n_neris  = round(total * injection_pct / 100)
    n_benign = total - n_neris
    neris_sample  = neris_df.sample(n=min(n_neris,  len(neris_df)),  random_state=seed, replace=True)
    benign_sample = benign_df.sample(n=min(n_benign, len(benign_df)), random_state=seed, replace=True)
    mix = pd.concat([neris_sample, benign_sample], ignore_index=True)
    return mix.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _compute_metrics(results_df, mix_df, injection_pct, total_time_s):
    results_df = results_df.copy()
    results_df["true_family"] = mix_df["family"].values
    results_df["is_neris"]    = (mix_df["family"].values == INJECTION_FAMILY).astype(int)
    results_df["is_benign"]   = (mix_df["family"].values == "Benign").astype(int)
    results_df["detected"]    = results_df["final_category"].isin(
                                    ["KNOWN_BOTNET", "NOVEL_BEHAVIOR"]).astype(int)
    n_neris  = results_df["is_neris"].sum()
    n_benign = results_df["is_benign"].sum()
    tpr = float(results_df.loc[results_df["is_neris"]  == 1, "detected"].mean()) if n_neris  > 0 else 0.0
    fpr = float(results_df.loc[results_df["is_benign"] == 1, "detected"].mean()) if n_benign > 0 else 0.0
    cat = results_df["final_category"].value_counts().to_dict()
    return {
        "injection_pct": injection_pct,
        "n_neris": int(n_neris), "n_benign": int(n_benign),
        "total_flows": len(results_df),
        "tpr": round(tpr, 4), "fpr": round(fpr, 4),
        "known_botnet_count":    cat.get("KNOWN_BOTNET",   0),
        "novel_behavior_count":  cat.get("NOVEL_BEHAVIOR", 0),
        "benign_count":          cat.get("BENIGN",         0),
        "uncertain_count":       cat.get("UNCERTAIN",      0),
        "total_time_s":          round(total_time_s, 2),
        "ms_per_flow":           round(total_time_s * 1000 / max(len(results_df), 1), 3),
    }


def _plot_results(summary_df):
    rates = summary_df["injection_pct"].values
    C_TPR = "#2ecc71"; C_FPR = "#e74c3c"
    C_KNOWN = "#3498db"; C_NOVEL = "#f39c12"; C_BEN = "#95a5a6"

    # Chart 1: TPR and FPR
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rates, summary_df["tpr"] * 100, "o-", color=C_TPR, lw=2.5, markersize=8, label="Detection Rate (TPR)")
    ax.plot(rates, summary_df["fpr"] * 100, "s--", color=C_FPR, lw=2, markersize=7, label="False Positive Rate (FPR)")
    ax.fill_between(rates, summary_df["tpr"] * 100, alpha=0.12, color=C_TPR)
    ax.fill_between(rates, summary_df["fpr"] * 100, alpha=0.08, color=C_FPR)
    ax.set_xlabel("Injection Rate (% Neris flows)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"BotnetLens Cascade — Detection vs FPR\nFamily: {INJECTION_FAMILY}  |  Cascade: LightGBM + IsolationForest", fontsize=11, fontweight="bold")
    ax.set_xticks(rates); ax.set_ylim(0, 105); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "injection_tpr_fpr.png", dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Chart -> %s", OUT_DIR / "injection_tpr_fpr.png")

    # Chart 2: Cascade breakdown stacked bar
    total = summary_df["total_flows"].values
    known  = summary_df["known_botnet_count"].values  / total * 100
    novel  = summary_df["novel_behavior_count"].values / total * 100
    benign = summary_df["benign_count"].values         / total * 100
    unc    = summary_df["uncertain_count"].values      / total * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.array(rates, dtype=float)
    w = 6
    ax.bar(x, known,  width=w, label="KNOWN_BOTNET",   color=C_KNOWN,  alpha=0.9)
    ax.bar(x, novel,  width=w, label="NOVEL_BEHAVIOR",  color=C_NOVEL,  alpha=0.9, bottom=known)
    ax.bar(x, benign, width=w, label="BENIGN",          color=C_BEN,    alpha=0.9, bottom=known+novel)
    ax.bar(x, unc,    width=w, label="UNCERTAIN",       color=C_FPR,    alpha=0.5, bottom=known+novel+benign)
    ax.set_xlabel("Injection Rate (% Neris flows)", fontsize=12, fontweight="bold")
    ax.set_ylabel("% of Total Flows", fontsize=12, fontweight="bold")
    ax.set_title(f"Cascade Decision Breakdown per Injection Rate\nFamily: {INJECTION_FAMILY}  |  Flows/run: {TOTAL_FLOWS:,}", fontsize=11, fontweight="bold")
    ax.set_xticks(rates); ax.set_ylim(0, 108); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "injection_cascade_breakdown.png", dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Chart -> %s", OUT_DIR / "injection_cascade_breakdown.png")

    # Chart 3: Latency
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(rates, summary_df["ms_per_flow"], width=6, color="#9b59b6", alpha=0.85, edgecolor="white")
    for r, v in zip(rates, summary_df["ms_per_flow"]):
        ax.text(r, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Injection Rate (% Neris flows)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Latency (ms/flow)", fontsize=12, fontweight="bold")
    ax.set_title("Cascade Inference Latency per Injection Rate", fontsize=11, fontweight="bold")
    ax.set_xticks(rates); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "injection_latency.png", dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Chart -> %s", OUT_DIR / "injection_latency.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Neris injection sweep (10% to 100%).")
    parser.add_argument("--total-flows",      type=int,   default=TOTAL_FLOWS)
    parser.add_argument("--seed",             type=int,   default=RANDOM_SEED)
    parser.add_argument("--binary-model",     default="lightgbm")
    parser.add_argument("--multiclass-model", default="lightgbm")
    parser.add_argument("--anomaly-model",    default="isolation_forest")
    parser.add_argument("--confidence",       type=float, default=0.70)
    parser.add_argument("--no-cache",         action="store_true",
                        help="Force reload pools from processed CSV even if cache exists")
    args = parser.parse_args(argv)

    logger.info("=" * 60)
    logger.info("BotnetLens Neris Injection Experiment  rates=%s  flows/run=%d", INJECTION_RATES, args.total_flows)

    engine = CascadeEngine(
        binary_model_name=args.binary_model,
        multiclass_model_name=args.multiclass_model,
        anomaly_model_name=args.anomaly_model,
        confidence_threshold=args.confidence,
    )
    try:
        engine.load()
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc); return 1

    neris_df, benign_df = _load_family_pools(args.seed, use_cache=not args.no_cache)
    if len(neris_df) < 100:
        logger.error("Neris pool too small (%d rows).", len(neris_df)); return 1

    all_metrics = []
    for rate in INJECTION_RATES:
        n_inj = round(args.total_flows * rate / 100)
        logger.info("--- Injection %d%%: %d Neris + %d Benign ---", rate, n_inj, args.total_flows - n_inj)
        mix_df = _build_mix(neris_df, benign_df, rate, args.total_flows, args.seed)
        t0 = time.time()
        results_df = engine.predict_batch(mix_df)
        elapsed = time.time() - t0
        results_df["true_family"] = mix_df["family"].values
        results_df.to_csv(OUT_DIR / f"run_{rate:03d}pct_predictions.csv", index=False)
        m = _compute_metrics(results_df, mix_df, rate, elapsed)
        all_metrics.append(m)
        logger.info("  TPR=%.1f%%  FPR=%.1f%%  KNOWN=%d  NOVEL=%d  %.1fs  %.3fms/flow",
                    m["tpr"]*100, m["fpr"]*100, m["known_botnet_count"], m["novel_behavior_count"],
                    elapsed, m["ms_per_flow"])

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(OUT_DIR / "injection_sweep_results.csv", index=False)
    logger.info("Results -> %s", OUT_DIR / "injection_sweep_results.csv")

    print("\n" + "=" * 70)
    print(f"{'Inj%':>5}  {'TPR%':>8}  {'FPR%':>8}  {'KNOWN':>7}  {'NOVEL':>7}  {'ms/flow':>9}")
    print("-" * 70)
    for m in all_metrics:
        print(f"{m['injection_pct']:>5}  {m['tpr']*100:>7.1f}%  {m['fpr']*100:>7.1f}%  "
              f"{m['known_botnet_count']:>7}  {m['novel_behavior_count']:>7}  {m['ms_per_flow']:>9.3f}")
    print("=" * 70)

    _plot_results(summary_df)
    logger.info("Injection experiment complete. Outputs -> %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
