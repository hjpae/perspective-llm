"""
ATS 05B — finish person-signal audit after the bootstrap-column bug
===================================================================

Use this if ats_05_person_signal_audit.py already reached:
    "Running same-language WRONG-PERSON support control..."
before crashing with:
    KeyError: 'Column not found: accuracy'

It reads the files that 05 already wrote and performs ONLY the final summary
and bootstrap stage. No Hugging Face streaming or kNN tuning is repeated.

Expected repo location:
    perspective-llm/ATS/evaluation/ats_05b_finish_person_signal_audit.py
(or ATS/eda/ if that is where you are keeping 05).
"""

from pathlib import Path
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
RESULT_ROOT = REPO_ROOT / "results" / "ATS" / "person_signal_audit"

N_BOOTSTRAP = 3000
BOOTSTRAP_SEED = 20260923


def summarize_predictions(pred_df: pd.DataFrame):
    out = []
    for model, g in pred_df.groupby("model"):
        out.append({
            "model": model,
            "n_query": len(g),
            "n_users": g["user_id"].nunique(),
            "accuracy": g["correct"].mean(),
            "nll": g["nll"].mean(),
            "brier": g["brier"].mean(),
            "mean_p_true": g["p_true"].mean(),
            "mean_neighbors_used": g["neighbors_used"].mean(),
        })
    return pd.DataFrame(out).sort_values("nll")


def clustered_bootstrap_delta(
    pred_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    n_boot=N_BOOTSTRAP,
):
    metric_col = {
        "accuracy": "correct",
        "nll": "nll",
        "brier": "brier",
    }[metric]

    a = (
        pred_df[pred_df["model"] == model_a]
        .groupby("user_id")[metric_col]
        .mean()
    )
    b = (
        pred_df[pred_df["model"] == model_b]
        .groupby("user_id")[metric_col]
        .mean()
    )

    common = a.index.intersection(b.index)
    diffs = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = np.empty(n_boot, dtype=float)
    n = len(diffs)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()

    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "n_users": n,
        "delta": float(diffs.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def bootstrap_real_vs_wrong(real_pred, wrong_user_df):
    real_user = (
        real_pred[real_pred["model"] == "B4_person_knn"]
        .groupby("user_id")
        .agg(
            real_accuracy=("correct", "mean"),
            real_nll=("nll", "mean"),
            real_brier=("brier", "mean"),
        )
        .reset_index()
    )
    m = real_user.merge(wrong_user_df, on="user_id", how="inner")

    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 10)

    for metric in ("accuracy", "nll", "brier"):
        diffs = (
            m[f"real_{metric}"].to_numpy()
            - m[f"wrong_{metric}"].to_numpy()
        )
        boots = np.empty(N_BOOTSTRAP)
        n = len(diffs)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            boots[i] = diffs[idx].mean()

        rows.append({
            "model_a": "B4_person_knn",
            "model_b": "B5_wrong_person_knn",
            "metric": metric,
            "n_users": n,
            "delta": float(diffs.mean()),
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
        })
    return rows


def main():
    test_pred_path = RESULT_ROOT / "test_predictions.csv"
    wrong_user_path = RESULT_ROOT / "wrong_person_per_user.csv"
    shuffle_path = RESULT_ROOT / "wrong_person_shuffle_summary.csv"

    missing = [p for p in (test_pred_path, wrong_user_path, shuffle_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "05 crashed before writing all prerequisite outputs. Missing:\\n"
            + "\\n".join(str(p) for p in missing)
            + "\\nIn that case run the fixed full 05 script instead."
        )

    test_pred = pd.read_csv(test_pred_path)
    wrong_user = pd.read_csv(wrong_user_path)
    shuffle_summary = pd.read_csv(shuffle_path)

    test_summary = summarize_predictions(test_pred)

    wrong_overall = {
        "model": "B5_wrong_person_knn",
        "n_query": int(shuffle_summary["n_query"].median()),
        "n_users": int(len(wrong_user)),
        "accuracy": float(shuffle_summary["accuracy"].mean()),
        "nll": float(shuffle_summary["nll"].mean()),
        "brier": float(shuffle_summary["brier"].mean()),
        "mean_p_true": np.nan,
        "mean_neighbors_used": float(shuffle_summary["mean_neighbors_used"].mean()),
    }

    summary = pd.concat(
        [test_summary, pd.DataFrame([wrong_overall])],
        ignore_index=True,
    ).sort_values("nll")

    summary.to_csv(
        RESULT_ROOT / "test_baseline_summary_with_wrong_person.csv",
        index=False,
    )

    delta_rows = []
    for metric in ("accuracy", "nll", "brier"):
        delta_rows.append(
            clustered_bootstrap_delta(
                test_pred, "B4_person_knn", "B2_lang_item", metric
            )
        )
        delta_rows.append(
            clustered_bootstrap_delta(
                test_pred, "B4_person_knn", "B3_position_profile", metric
            )
        )

    delta_rows.extend(bootstrap_real_vs_wrong(test_pred, wrong_user))
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(RESULT_ROOT / "bootstrap_deltas.csv", index=False)

    print("\\n" + "=" * 78)
    print("PERSON-SPECIFIC PREDICTIVE SIGNAL AUDIT — TEST")
    print("=" * 78)
    print(
        summary[
            ["model", "n_query", "n_users", "accuracy", "nll", "brier",
             "mean_neighbors_used"]
        ].to_string(index=False)
    )

    print("\\n[key user-cluster bootstrap deltas: model_a - model_b]")
    print(delta_df.to_string(index=False))

    print("\\nInterpretation:")
    print("  Accuracy: positive B4-B2 / B4-B5 is good.")
    print("  NLL/Brier: negative B4-B2 / B4-B5 is good.")
    print("\\nWrote:", RESULT_ROOT.resolve())


if __name__ == "__main__":
    main()
