"""
ATS / Community Alignment deep EDA
==================================

Purpose
-------
Deep, full-corpus census of facebook/community-alignment-dataset before ATS
modeling. This script streams the complete Hugging Face dataset and stores only
small aggregate tables -- NOT the 1.9 GB raw corpus.

It audits:
  1) users, waves, and cross-wave overlap
  2) language and prompt-origin structure
  3) conversation-length / missing-turn patterns
  4) preference choice-position distributions
  5) explanation/feedback coverage and lengths
  6) candidate-response length / position confounds
  7) repeated-prompt and repeated-comparison structure
  8) within-prompt preference heterogeneity
  9) annotator-level choice-position bias / entropy
 10) heavy-user tails and duplicate-prompt exposure
 11) data-integrity / metadata-consistency checks

Important interpretation constraints
------------------------------------
* Community Alignment has only a coarse `wave` field. Conversation IDs are
  treated as identifiers, NOT timestamps.
* Within-wave ordering is therefore UNKNOWN.
* Sensitive demographic fields are deliberately excluded from this EDA and
  should not be ATS model inputs.
* Prompt / response texts are not written to disk except short previews for a
  small top-repeated-prompt report.

Expected location:
    perspective-llm/ATS/eda/ats_04_ca_deep_eda.py

Run from Spyder normally. Paths are anchored to the repository root and are
independent of Spyder's current working directory.

Dependencies:
    pip install datasets huggingface_hub pandas numpy
"""

# %% Imports
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import get_token


# %% Paths / config
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
RESULT_ROOT = REPO_ROOT / "results" / "ATS" / "eda" / "community_alignment_deep"

HF_DATASET = "facebook/community-alignment-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN") or get_token()

PROGRESS_EVERY = 5000
TOP_PROMPTS_TO_SAVE = 250
TOP_USERS_TO_SAVE = 100

TURNS = ("first", "second", "third", "fourth")
CHOICES = ("a", "b", "c", "d")


# %% Utilities
def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_text(x: Any) -> str:
    s = clean_text(x).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", clean_text(text)))


def normalize_choice(x: Any) -> str:
    s = clean_text(x).lower().strip()
    if not s:
        return ""
    exact = {
        "a": "a", "b": "b", "c": "c", "d": "d",
        "response_a": "a", "response_b": "b",
        "response_c": "c", "response_d": "d",
        "option_a": "a", "option_b": "b",
        "option_c": "c", "option_d": "d",
        "response a": "a", "response b": "b",
        "response c": "c", "response d": "d",
        "option a": "a", "option b": "b",
        "option c": "c", "option d": "d",
    }
    if s in exact:
        return exact[s]

    m = re.search(r"(?:response|option)?[\s_\-:]*([abcd])\b", s)
    if m:
        return m.group(1)

    # Sometimes fields may contain only a capital letter plus punctuation.
    m = re.fullmatch(r"\s*([abcd])[\.\):]?\s*", s)
    return m.group(1) if m else ""


def entropy_from_counter(c: Counter) -> float:
    total = sum(c.values())
    if total <= 0:
        return float("nan")
    probs = [v / total for v in c.values() if v > 0]
    return float(-sum(p * math.log2(p) for p in probs))


def max_share(c: Counter) -> float:
    total = sum(c.values())
    return float(max(c.values()) / total) if total else float("nan")


def write_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# %% Aggregator containers
class DeepAudit:
    def __init__(self):
        self.n_rows = 0

        # Schema / missingness
        self.column_seen = Counter()
        self.column_missing = Counter()

        # User / wave / language / country / origin
        self.users = set()
        self.waves = set()
        self.wave_users = defaultdict(set)
        self.wave_rows = Counter()

        self.lang_rows = Counter()
        self.lang_users = defaultdict(set)
        self.wave_lang_rows = Counter()

        self.country_rows = Counter()
        self.country_users = defaultdict(set)

        self.origin_rows = Counter()              # pregenerated / user_initiated
        self.wave_origin_rows = Counter()
        self.lang_origin_rows = Counter()

        # Per-user summaries
        self.user_rows = Counter()
        self.user_waves = defaultdict(set)
        self.user_langs = defaultdict(set)
        self.user_countries = defaultdict(set)
        self.user_origin = defaultdict(Counter)
        self.user_turn_counts = Counter()
        self.user_feedback_counts = Counter()
        self.user_choice_counts = defaultdict(Counter)
        self.user_chosen_longest = Counter()
        self.user_chosen_shortest = Counter()
        self.user_length_comparisons = Counter()
        self.user_prompt_hashes = defaultdict(set)
        self.user_duplicate_prompt_rows = Counter()

        # Conversation turn structure
        self.turn_presence = Counter()
        self.turn_preference_present = Counter()
        self.turn_feedback_present = Counter()
        self.turn_feedback_words = defaultdict(list)
        self.turn_response_complete = Counter()
        self.turn_missing_pattern = Counter()
        self.turn_count_dist = Counter()

        # Preference choice position
        self.choice_counts = Counter()
        self.choice_by_turn = Counter()
        self.choice_by_wave = Counter()
        self.choice_by_lang = Counter()
        self.choice_by_origin = Counter()
        self.unparsed_preferred = Counter()

        # Candidate response lengths / potential position confounds
        self.response_word_sums = Counter()
        self.response_word_ns = Counter()
        self.response_char_sums = Counter()
        self.response_char_ns = Counter()
        self.chosen_word_lengths = []
        self.rejected_word_lengths = []
        self.chosen_is_longest = 0
        self.chosen_is_shortest = 0
        self.length_comparison_n = 0

        # First-turn prompt overlap (main shared-prompt structure)
        self.prompt_stats = {}  # prompt_hash -> dict
        self.comparison_stats = {}  # prompt+ordered candidates -> dict

        # All-turn prompt duplication
        self.all_prompt_counts = Counter()
        self.prompt_turn_counts = Counter()

        # Metadata consistency
        self.user_lang_inconsistency = set()
        self.user_country_inconsistency = set()

    def _prompt_record(self, h: str, preview: str):
        if h not in self.prompt_stats:
            self.prompt_stats[h] = {
                "n_rows": 0,
                "users": set(),
                "choices": Counter(),
                "waves": Counter(),
                "langs": Counter(),
                "origins": Counter(),
                "ordered_candidate_hashes": Counter(),
                "preview": preview[:180],
            }
        return self.prompt_stats[h]

    def _comparison_record(self, h: str):
        if h not in self.comparison_stats:
            self.comparison_stats[h] = {
                "n_rows": 0,
                "users": set(),
                "choices": Counter(),
                "waves": Counter(),
                "langs": Counter(),
            }
        return self.comparison_stats[h]

    def add(self, row: Dict[str, Any]):
        self.n_rows += 1

        # Schema / missingness based on actual streamed rows.
        for k, v in row.items():
            self.column_seen[k] += 1
            if clean_text(v) == "":
                self.column_missing[k] += 1

        uid = clean_text(row.get("annotator_id"))
        wave = clean_text(row.get("wave"))
        lang = clean_text(row.get("assigned_lang"))
        country = clean_text(row.get("annotator_country"))
        pregenerated = row.get("is_pregenerated_first_prompt")
        origin = "pregenerated" if bool(pregenerated) else "user_initiated"

        if uid:
            self.users.add(uid)
            self.user_rows[uid] += 1
            if wave:
                self.user_waves[uid].add(wave)
            if lang:
                self.user_langs[uid].add(lang)
            if country:
                self.user_countries[uid].add(country)
            self.user_origin[uid][origin] += 1

        if wave:
            self.waves.add(wave)
            self.wave_rows[wave] += 1
            if uid:
                self.wave_users[wave].add(uid)

        if lang:
            self.lang_rows[lang] += 1
            if uid:
                self.lang_users[lang].add(uid)
            if wave:
                self.wave_lang_rows[(wave, lang)] += 1

        if country:
            self.country_rows[country] += 1
            if uid:
                self.country_users[country].add(uid)

        self.origin_rows[origin] += 1
        if wave:
            self.wave_origin_rows[(wave, origin)] += 1
        if lang:
            self.lang_origin_rows[(lang, origin)] += 1

        # Track metadata consistency.
        if uid and len(self.user_langs[uid]) > 1:
            self.user_lang_inconsistency.add(uid)
        if uid and len(self.user_countries[uid]) > 1:
            self.user_country_inconsistency.add(uid)

        # Turn-level structure
        present_pattern = []
        n_present_turns = 0

        for turn_idx, turn in enumerate(TURNS, 1):
            prompt = clean_text(row.get(f"{turn}_turn_prompt"))
            preferred_raw = row.get(f"{turn}_turn_preferred_response")
            preferred = normalize_choice(preferred_raw)
            feedback = clean_text(row.get(f"{turn}_turn_feedback"))

            candidates = {
                c: clean_text(row.get(f"{turn}_turn_response_{c}"))
                for c in CHOICES
            }

            prompt_present = bool(prompt)
            present_pattern.append("1" if prompt_present else "0")
            if prompt_present:
                n_present_turns += 1
                self.turn_presence[turn] += 1
                if uid:
                    self.user_turn_counts[uid] += 1

                ph = stable_hash(normalize_text(prompt))
                self.all_prompt_counts[ph] += 1
                self.prompt_turn_counts[(turn, ph)] += 1

                if uid:
                    if ph in self.user_prompt_hashes[uid]:
                        self.user_duplicate_prompt_rows[uid] += 1
                    self.user_prompt_hashes[uid].add(ph)

            if clean_text(preferred_raw):
                self.turn_preference_present[turn] += 1

            if preferred:
                self.choice_counts[preferred] += 1
                self.choice_by_turn[(turn, preferred)] += 1
                if wave:
                    self.choice_by_wave[(wave, preferred)] += 1
                if lang:
                    self.choice_by_lang[(lang, preferred)] += 1
                self.choice_by_origin[(origin, preferred)] += 1
                if uid:
                    self.user_choice_counts[uid][preferred] += 1
            elif clean_text(preferred_raw):
                self.unparsed_preferred[clean_text(preferred_raw)[:80]] += 1

            if feedback:
                self.turn_feedback_present[turn] += 1
                self.turn_feedback_words[turn].append(word_count(feedback))
                if uid:
                    self.user_feedback_counts[uid] += 1

            complete_candidates = all(bool(candidates[c]) for c in CHOICES)
            if complete_candidates:
                self.turn_response_complete[turn] += 1

            # Candidate-length position audit.
            for c in CHOICES:
                if candidates[c]:
                    wc = word_count(candidates[c])
                    self.response_word_sums[(turn, c)] += wc
                    self.response_word_ns[(turn, c)] += 1
                    self.response_char_sums[(turn, c)] += len(candidates[c])
                    self.response_char_ns[(turn, c)] += 1

            if preferred and complete_candidates:
                lengths = {c: word_count(candidates[c]) for c in CHOICES}
                chosen_len = lengths[preferred]
                rejected = [lengths[c] for c in CHOICES if c != preferred]
                self.chosen_word_lengths.append(chosen_len)
                self.rejected_word_lengths.extend(rejected)
                self.length_comparison_n += 1
                if chosen_len == max(lengths.values()):
                    self.chosen_is_longest += 1
                    if uid:
                        self.user_chosen_longest[uid] += 1
                if chosen_len == min(lengths.values()):
                    self.chosen_is_shortest += 1
                    if uid:
                        self.user_chosen_shortest[uid] += 1
                if uid:
                    self.user_length_comparisons[uid] += 1

            # First-turn repeated-prompt / repeated-comparison audit.
            if turn == "first" and prompt:
                ph = stable_hash(normalize_text(prompt))
                rec = self._prompt_record(ph, prompt)
                rec["n_rows"] += 1
                if uid:
                    rec["users"].add(uid)
                if preferred:
                    rec["choices"][preferred] += 1
                if wave:
                    rec["waves"][wave] += 1
                if lang:
                    rec["langs"][lang] += 1
                rec["origins"][origin] += 1

                ordered_candidate_text = "\n<SEP>\n".join(
                    normalize_text(candidates[c]) for c in CHOICES
                )
                ordered_hash = stable_hash(ordered_candidate_text)
                rec["ordered_candidate_hashes"][ordered_hash] += 1

                comparison_hash = stable_hash(ph + "|" + ordered_hash)
                crec = self._comparison_record(comparison_hash)
                crec["n_rows"] += 1
                if uid:
                    crec["users"].add(uid)
                if preferred:
                    crec["choices"][preferred] += 1
                if wave:
                    crec["waves"][wave] += 1
                if lang:
                    crec["langs"][lang] += 1

        self.turn_missing_pattern["".join(present_pattern)] += 1
        self.turn_count_dist[n_present_turns] += 1


# %% Serialization
def save_counter(counter: Counter, columns: Tuple[str, ...], filename: str):
    rows = []
    for key, count in counter.items():
        if isinstance(key, tuple):
            vals = list(key)
        else:
            vals = [key]
        row = {columns[i]: vals[i] for i in range(len(columns))}
        row["count"] = count
        rows.append(row)
    pd.DataFrame(rows).sort_values(
        "count", ascending=False, kind="stable"
    ).to_csv(RESULT_ROOT / filename, index=False)


def finalize(a: DeepAudit):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    # Basic summary
    summary = {
        "rows": a.n_rows,
        "unique_users": len(a.users),
        "waves": sorted(a.waves),
        "unique_waves": len(a.waves),
        "hf_authenticated": bool(HF_TOKEN),
        "user_language_inconsistency_count": len(a.user_lang_inconsistency),
        "user_country_inconsistency_count": len(a.user_country_inconsistency),
        "unparsed_preferred_labels": dict(a.unparsed_preferred),
        "length_comparisons": a.length_comparison_n,
        "fraction_chosen_longest": (
            a.chosen_is_longest / a.length_comparison_n
            if a.length_comparison_n else None
        ),
        "fraction_chosen_shortest": (
            a.chosen_is_shortest / a.length_comparison_n
            if a.length_comparison_n else None
        ),
        "mean_chosen_response_words": (
            float(np.mean(a.chosen_word_lengths))
            if a.chosen_word_lengths else None
        ),
        "mean_rejected_response_words": (
            float(np.mean(a.rejected_word_lengths))
            if a.rejected_word_lengths else None
        ),
    }
    write_json(summary, RESULT_ROOT / "ca_deep_summary.json")

    # Missingness
    miss_rows = []
    for col in sorted(a.column_seen):
        seen = a.column_seen[col]
        missing = a.column_missing[col]
        miss_rows.append({
            "column": col,
            "rows_seen": seen,
            "missing": missing,
            "missing_fraction": missing / seen if seen else np.nan,
        })
    pd.DataFrame(miss_rows).to_csv(
        RESULT_ROOT / "schema_missingness.csv", index=False
    )

    # Wave overlap / membership
    waves = sorted(a.waves)
    overlap_rows = []
    for i, w1 in enumerate(waves):
        for w2 in waves[i:]:
            s1 = a.wave_users[w1]
            s2 = a.wave_users[w2]
            inter = s1 & s2
            union = s1 | s2
            overlap_rows.append({
                "wave_a": w1,
                "wave_b": w2,
                "users_a": len(s1),
                "users_b": len(s2),
                "overlap_users": len(inter),
                "jaccard": len(inter) / len(union) if union else np.nan,
            })
    pd.DataFrame(overlap_rows).to_csv(
        RESULT_ROOT / "wave_user_overlap.csv", index=False
    )

    membership = Counter(tuple(sorted(ws)) for ws in a.user_waves.values())
    membership_rows = [{
        "wave_membership": "|".join(ws),
        "users": n,
    } for ws, n in membership.items()]
    pd.DataFrame(membership_rows).sort_values(
        "users", ascending=False
    ).to_csv(RESULT_ROOT / "user_wave_membership.csv", index=False)

    # User-level table
    user_rows = []
    for uid in sorted(a.users):
        cc = a.user_choice_counts[uid]
        n_choices = sum(cc.values())
        n_len = a.user_length_comparisons[uid]
        user_rows.append({
            "user_id": uid,
            "conversations": a.user_rows[uid],
            "waves": "|".join(sorted(a.user_waves[uid])),
            "n_waves": len(a.user_waves[uid]),
            "languages": "|".join(sorted(a.user_langs[uid])),
            "n_languages": len(a.user_langs[uid]),
            "countries": "|".join(sorted(a.user_countries[uid])),
            "n_countries": len(a.user_countries[uid]),
            "pregenerated_conversations": a.user_origin[uid]["pregenerated"],
            "user_initiated_conversations": a.user_origin[uid]["user_initiated"],
            "observed_turns": a.user_turn_counts[uid],
            "feedback_items": a.user_feedback_counts[uid],
            "preference_choices": n_choices,
            "choice_entropy_bits": entropy_from_counter(cc),
            "dominant_choice_share": max_share(cc),
            "choice_a": cc["a"],
            "choice_b": cc["b"],
            "choice_c": cc["c"],
            "choice_d": cc["d"],
            "unique_prompt_hashes": len(a.user_prompt_hashes[uid]),
            "duplicate_prompt_rows_within_user": a.user_duplicate_prompt_rows[uid],
            "length_comparisons": n_len,
            "fraction_chosen_longest": (
                a.user_chosen_longest[uid] / n_len if n_len else np.nan
            ),
            "fraction_chosen_shortest": (
                a.user_chosen_shortest[uid] / n_len if n_len else np.nan
            ),
        })
    users_df = pd.DataFrame(user_rows)
    users_df.to_csv(RESULT_ROOT / "user_level_census.csv", index=False)

    users_df.nlargest(TOP_USERS_TO_SAVE, "conversations").to_csv(
        RESULT_ROOT / "heavy_users_top100.csv", index=False
    )

    # Basic categorical counts
    save_counter(a.wave_rows, ("wave",), "wave_rows.csv")
    save_counter(a.lang_rows, ("language",), "language_rows.csv")
    save_counter(a.country_rows, ("country",), "country_rows.csv")
    save_counter(a.origin_rows, ("origin",), "prompt_origin_rows.csv")
    save_counter(a.wave_lang_rows, ("wave", "language"), "wave_language_rows.csv")
    save_counter(a.wave_origin_rows, ("wave", "origin"), "wave_prompt_origin_rows.csv")
    save_counter(a.lang_origin_rows, ("language", "origin"), "language_prompt_origin_rows.csv")

    save_counter(a.turn_presence, ("turn",), "turn_prompt_presence.csv")
    save_counter(a.turn_preference_present, ("turn",), "turn_preference_presence.csv")
    save_counter(a.turn_feedback_present, ("turn",), "turn_feedback_presence.csv")
    save_counter(a.turn_response_complete, ("turn",), "turn_complete_candidates.csv")
    save_counter(a.turn_missing_pattern, ("turn_presence_pattern",), "turn_presence_patterns.csv")
    save_counter(a.turn_count_dist, ("n_turns",), "conversation_turn_count.csv")

    save_counter(a.choice_counts, ("choice",), "choice_position_global.csv")
    save_counter(a.choice_by_turn, ("turn", "choice"), "choice_position_by_turn.csv")
    save_counter(a.choice_by_wave, ("wave", "choice"), "choice_position_by_wave.csv")
    save_counter(a.choice_by_lang, ("language", "choice"), "choice_position_by_language.csv")
    save_counter(a.choice_by_origin, ("origin", "choice"), "choice_position_by_prompt_origin.csv")

    # Feedback lengths
    feedback_rows = []
    for turn, vals in a.turn_feedback_words.items():
        if vals:
            arr = np.asarray(vals)
            feedback_rows.append({
                "turn": turn,
                "n_feedback": len(arr),
                "mean_words": float(np.mean(arr)),
                "median_words": float(np.median(arr)),
                "p90_words": float(np.quantile(arr, .9)),
                "p99_words": float(np.quantile(arr, .99)),
            })
    pd.DataFrame(feedback_rows).to_csv(
        RESULT_ROOT / "feedback_length_summary.csv", index=False
    )

    # Candidate response length by option / turn
    length_rows = []
    for turn in TURNS:
        for c in CHOICES:
            n = a.response_word_ns[(turn, c)]
            length_rows.append({
                "turn": turn,
                "choice_position": c,
                "n": n,
                "mean_words": (
                    a.response_word_sums[(turn, c)] / n if n else np.nan
                ),
                "mean_chars": (
                    a.response_char_sums[(turn, c)]
                    / a.response_char_ns[(turn, c)]
                    if a.response_char_ns[(turn, c)] else np.nan
                ),
            })
    pd.DataFrame(length_rows).to_csv(
        RESULT_ROOT / "candidate_length_by_position.csv", index=False
    )

    # Repeated first-turn prompts
    prompt_rows = []
    for ph, rec in a.prompt_stats.items():
        prompt_rows.append({
            "prompt_hash": ph,
            "n_rows": rec["n_rows"],
            "n_users": len(rec["users"]),
            "n_ordered_candidate_sets": len(rec["ordered_candidate_hashes"]),
            "choice_entropy_bits": entropy_from_counter(rec["choices"]),
            "dominant_choice_share": max_share(rec["choices"]),
            "choice_a": rec["choices"]["a"],
            "choice_b": rec["choices"]["b"],
            "choice_c": rec["choices"]["c"],
            "choice_d": rec["choices"]["d"],
            "waves": "|".join(sorted(rec["waves"])),
            "languages": "|".join(sorted(rec["langs"])),
            "pregenerated_rows": rec["origins"]["pregenerated"],
            "user_initiated_rows": rec["origins"]["user_initiated"],
            "prompt_preview": rec["preview"],
        })
    prompt_df = pd.DataFrame(prompt_rows)
    prompt_df.to_csv(RESULT_ROOT / "first_turn_prompt_overlap.csv", index=False)

    prompt_df.sort_values(
        ["n_users", "n_rows"], ascending=False
    ).head(TOP_PROMPTS_TO_SAVE).to_csv(
        RESULT_ROOT / "top_repeated_first_turn_prompts.csv", index=False
    )

    # Exact prompt + ordered candidate-set overlap
    comp_rows = []
    for ch, rec in a.comparison_stats.items():
        comp_rows.append({
            "comparison_hash": ch,
            "n_rows": rec["n_rows"],
            "n_users": len(rec["users"]),
            "choice_entropy_bits": entropy_from_counter(rec["choices"]),
            "dominant_choice_share": max_share(rec["choices"]),
            "choice_a": rec["choices"]["a"],
            "choice_b": rec["choices"]["b"],
            "choice_c": rec["choices"]["c"],
            "choice_d": rec["choices"]["d"],
            "waves": "|".join(sorted(rec["waves"])),
            "languages": "|".join(sorted(rec["langs"])),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(
        RESULT_ROOT / "exact_comparison_overlap.csv", index=False
    )

    # High-level overlap diagnostics
    overlap_diag = {
        "first_turn_unique_prompt_hashes": int(len(prompt_df)),
        "first_turn_prompts_with_2plus_users": int((prompt_df["n_users"] >= 2).sum()),
        "first_turn_prompts_with_10plus_users": int((prompt_df["n_users"] >= 10).sum()),
        "first_turn_prompts_with_2plus_candidate_sets": int(
            (prompt_df["n_ordered_candidate_sets"] >= 2).sum()
        ),
        "exact_comparisons_total": int(len(comp_df)),
        "exact_comparisons_with_2plus_users": int((comp_df["n_users"] >= 2).sum()),
        "exact_comparisons_with_10plus_users": int((comp_df["n_users"] >= 10).sum()),
        "median_users_per_repeated_prompt": (
            float(prompt_df.loc[prompt_df["n_users"] >= 2, "n_users"].median())
            if (prompt_df["n_users"] >= 2).any() else None
        ),
        "median_choice_entropy_repeated_exact_comparison": (
            float(comp_df.loc[comp_df["n_users"] >= 2, "choice_entropy_bits"].median())
            if (comp_df["n_users"] >= 2).any() else None
        ),
    }
    write_json(overlap_diag, RESULT_ROOT / "prompt_overlap_diagnostics.json")

    # Data-integrity flags
    noncontiguous_patterns = []
    for pattern, n in a.turn_missing_pattern.items():
        # Valid contiguous pattern is some 1s followed by some 0s, e.g. 1100.
        if not re.fullmatch(r"1+0*", pattern):
            noncontiguous_patterns.append({"pattern": pattern, "count": n})

    integrity = {
        "users_with_multiple_assigned_languages": len(a.user_lang_inconsistency),
        "users_with_multiple_countries": len(a.user_country_inconsistency),
        "noncontiguous_turn_presence_patterns": noncontiguous_patterns,
        "unparsed_preferred_labels": dict(a.unparsed_preferred),
    }
    write_json(integrity, RESULT_ROOT / "integrity_flags.json")

    # Console report
    print("\n" + "=" * 78)
    print("COMMUNITY ALIGNMENT DEEP EDA")
    print("=" * 78)
    print(f"rows:  {a.n_rows:,}")
    print(f"users: {len(a.users):,}")
    print(f"waves: {sorted(a.waves)}")

    print("\n[user wave membership]")
    for membership_key, n in membership.most_common():
        print(f"  {membership_key}: {n:,}")

    print("\n[rows by wave]")
    for k, v in a.wave_rows.most_common():
        print(f"  wave {k}: {v:,}")

    print("\n[prompt origin]")
    for k, v in a.origin_rows.most_common():
        print(f"  {k}: {v:,} ({v / a.n_rows:.1%})")

    print("\n[conversation turn count]")
    for k, v in sorted(a.turn_count_dist.items()):
        print(f"  {k} turns: {v:,} ({v / a.n_rows:.1%})")

    print("\n[choice position]")
    total_choice = sum(a.choice_counts.values())
    for k in CHOICES:
        v = a.choice_counts[k]
        print(f"  {k.upper()}: {v:,} ({v / total_choice:.1%})")

    print("\n[length confound]")
    print(f"  comparisons with 4 candidates: {a.length_comparison_n:,}")
    if a.length_comparison_n:
        print(f"  chosen is longest: {a.chosen_is_longest / a.length_comparison_n:.1%}")
        print(f"  chosen is shortest: {a.chosen_is_shortest / a.length_comparison_n:.1%}")
        print(f"  mean chosen words:   {np.mean(a.chosen_word_lengths):.2f}")
        print(f"  mean rejected words: {np.mean(a.rejected_word_lengths):.2f}")

    print("\n[prompt overlap]")
    for k, v in overlap_diag.items():
        print(f"  {k}: {v}")

    print("\n[integrity]")
    print(f"  users with multiple languages: {len(a.user_lang_inconsistency)}")
    print(f"  users with multiple countries: {len(a.user_country_inconsistency)}")
    print(f"  noncontiguous turn patterns: {noncontiguous_patterns}")
    print(f"  unparsed preferred labels: {dict(a.unparsed_preferred)}")

    print("\nWrote:", RESULT_ROOT.resolve())


# %% Main
def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("REPO_ROOT:   ", REPO_ROOT)
    print("RESULT_ROOT: ", RESULT_ROOT)
    print("HF auth:     ", "yes" if HF_TOKEN else "no (public streaming)")
    print("\nStreaming FULL Community Alignment corpus...")
    print("Raw rows are NOT persisted locally.\n")

    ds = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    audit = DeepAudit()

    for i, row in enumerate(ds, 1):
        audit.add(dict(row))
        if i % PROGRESS_EVERY == 0:
            print(
                f"  {i:,} rows | "
                f"users={len(audit.users):,} | "
                f"first-turn prompts={len(audit.prompt_stats):,}"
            )

    finalize(audit)


if __name__ == "__main__":
    main()
