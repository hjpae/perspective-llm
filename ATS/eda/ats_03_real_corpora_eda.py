"""EDA over the canonical ATS real-human corpora.

Robust to Spyder's `%runfile ... --wdir`: all project paths are anchored to the
repository root inferred from this file's location, not the current working directory.
"""
# %% Imports
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% PROJECT PATHS
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # .../perspective-llm

CANON_ROOT = REPO_ROOT / "data" / "ATS" / "canonical"
RESULT_ROOT = REPO_ROOT / "results" / "ATS" / "eda" / "real_corpora"

print("REPO_ROOT:  ", REPO_ROOT)
print("CANON_ROOT: ", CANON_ROOT)
print("RESULT_ROOT:", RESULT_ROOT)

SOURCES = ["thoughttrace", "community_alignment", "talk2ai"]

# %% Helpers
def read_table(folder: Path, name: str) -> pd.DataFrame:
    pq = folder / f"{name}.parquet"
    csv = folder / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, keep_default_na=False)
    return pd.DataFrame()

def q(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    vals = s.quantile([0, .25, .5, .75, .9, .99, 1.0])
    return {str(k): float(v) for k, v in vals.items()}

# %% Source audit
def audit_source(source: str):
    folder = CANON_ROOT / source
    users = read_table(folder, "users")
    episodes = read_table(folder, "episodes")
    events = read_table(folder, "events")
    signals = read_table(folder, "signals")

    if users.empty and episodes.empty and events.empty:
        print(f"[skip] {source}: no canonical tables found under {folder}")
        return None, None, None

    user_eps = episodes.groupby("user_id").size() if not episodes.empty else pd.Series(dtype=float)
    user_events = events.groupby("user_id").size() if not events.empty else pd.Series(dtype=float)
    episode_events = events.groupby("episode_id").size() if not events.empty else pd.Series(dtype=float)

    row = {
        "source": source,
        "users": int(users["user_id"].nunique()) if not users.empty else int(events["user_id"].nunique()),
        "episodes": len(episodes),
        "events": len(events),
        "signals": len(signals),
        "median_episodes_per_user": float(user_eps.median()) if len(user_eps) else np.nan,
        "p90_episodes_per_user": float(user_eps.quantile(.9)) if len(user_eps) else np.nan,
        "median_events_per_user": float(user_events.median()) if len(user_events) else np.nan,
        "median_events_per_episode": float(episode_events.median()) if len(episode_events) else np.nan,
        "users_with_2plus_episodes": int((user_eps >= 2).sum()) if len(user_eps) else 0,
        "users_with_4plus_episodes": int((user_eps >= 4).sum()) if len(user_eps) else 0,
        "users_with_10plus_episodes": int((user_eps >= 10).sum()) if len(user_eps) else 0,
    }

    roles = (
        events["role"].value_counts().rename_axis("role").reset_index(name="count")
        if not events.empty else pd.DataFrame()
    )
    if not roles.empty:
        roles["source"] = source

    stypes = (
        signals["signal_type"].value_counts().rename_axis("signal_type").reset_index(name="count")
        if not signals.empty else pd.DataFrame()
    )
    if not stypes.empty:
        stypes["source"] = source

    trajectories = pd.DataFrame({
        "source": source,
        "user_id": user_eps.index.astype(str),
        "episodes": user_eps.values,
        "events": user_events.reindex(user_eps.index).fillna(0).values,
    }) if len(user_eps) else pd.DataFrame()

    return row, (roles, stypes), trajectories

# %% Main
def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    if not CANON_ROOT.exists():
        raise FileNotFoundError(
            f"Canonical corpus folder not found:\n{CANON_ROOT}\n"
            "Run ATS/data_pipeline/ats_02_prepare_real_corpora.py first."
        )

    summary_rows = []
    role_frames = []
    signal_frames = []
    trajectory_frames = []

    for source in SOURCES:
        row, dist, traj = audit_source(source)
        if row is None:
            continue
        summary_rows.append(row)
        roles, stypes = dist
        if not roles.empty:
            role_frames.append(roles)
        if not stypes.empty:
            signal_frames.append(stypes)
        if not traj.empty:
            trajectory_frames.append(traj)

    summary = pd.DataFrame(summary_rows)
    roles = pd.concat(role_frames, ignore_index=True) if role_frames else pd.DataFrame()
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    trajectories = pd.concat(trajectory_frames, ignore_index=True) if trajectory_frames else pd.DataFrame()

    summary.to_csv(RESULT_ROOT / "corpus_summary.csv", index=False)
    roles.to_csv(RESULT_ROOT / "role_counts.csv", index=False)
    signals.to_csv(RESULT_ROOT / "signal_type_counts.csv", index=False)
    trajectories.to_csv(RESULT_ROOT / "user_trajectory_lengths.csv", index=False)

    print("\n[corpus summary]")
    print(summary.to_string(index=False))

    if summary.empty:
        raise RuntimeError(
            "No corpora were found. Check the printed CANON_ROOT path and verify "
            "that thoughttrace/ and community_alignment/ exist there."
        )

    if not trajectories.empty:
        print("\n[trajectory quantiles]")
        for source, sdf in trajectories.groupby("source"):
            print(source)
            print("  episodes/user:", q(sdf["episodes"]))
            print("  events/user:  ", q(sdf["events"]))

        plt.figure(figsize=(9, 5))
        for source, sdf in trajectories.groupby("source"):
            vals = np.sort(pd.to_numeric(sdf["episodes"], errors="coerce").dropna().values)
            if len(vals):
                y = np.arange(1, len(vals) + 1) / len(vals)
                plt.step(vals, y, where="post", label=source)
        plt.xlabel("Episodes per user")
        plt.ylabel("Empirical CDF")
        plt.title("Longitudinal coverage across real-human corpora")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / "episodes_per_user_ecdf.png", dpi=160)
        plt.close()

    print("\nWrote:", RESULT_ROOT.resolve())

if __name__ == "__main__":
    main()
