"""Shared adapters for ATS real-human corpora.

The canonical representation is deliberately ontology-light:

users.parquet
    source, user_id, metadata_json

episodes.parquet
    source, user_id, episode_id, start_time, end_time, model, language,
    topic, wave, order_basis, metadata_json

events.parquet
    source, user_id, episode_id, event_id, event_index, timestamp,
    role, text, metadata_json

signals.parquet
    source, user_id, episode_id, event_id, signal_type, signal_text,
    signal_label, signal_value, metadata_json

Important: signals are supervision/probe material, not ATS state ontology.
Downstream learned-state input should normally use events only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd


CANONICAL_TABLES = ("users", "episodes", "events", "signals")


def _json(x: Any) -> str:
    return json.dumps(x if x is not None else {}, ensure_ascii=False, sort_keys=True)


def _text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _coalesce(d: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _timestamp(x: Any) -> str:
    s = _text(x)
    return s


def _participant_from_thoughttrace(row: Dict[str, Any]) -> str:
    direct = _coalesce(row, "participant_id", "user_id", default="")
    if direct:
        return _text(direct)
    rid = _text(_coalesce(row, "id", "conversation_id", default=""))
    m = re.match(r"(user\d+)", rid)
    return m.group(1) if m else rid


def adapt_thoughttrace_rows(rows: Iterable[Dict[str, Any]], source: str = "thoughttrace") -> Dict[str, pd.DataFrame]:
    users: Dict[str, Dict[str, Any]] = {}
    episodes: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    signals: List[Dict[str, Any]] = []

    for row in rows:
        row = dict(row)
        user_id = _participant_from_thoughttrace(row)
        episode_id = _text(_coalesce(row, "id", "conversation_id", default=f"{user_id}_episode"))
        model = _text(_coalesce(row, "model_name", "model", default=""))
        created = _timestamp(_coalesce(row, "created_at", default=""))
        updated = _timestamp(_coalesce(row, "updated_at", default=""))

        # Privacy-minimal by default: retain task-level metadata, not demographics.
        users.setdefault(user_id, {
            "source": source,
            "user_id": user_id,
            "metadata_json": _json({}),
        })

        episodes.append({
            "source": source,
            "user_id": user_id,
            "episode_id": episode_id,
            "start_time": created,
            "end_time": updated,
            "model": model,
            "language": "en",
            "topic": "",
            "wave": "",
            "order_basis": "timestamp",
            "metadata_json": _json({
                "task_summary": row.get("task_summary", ""),
                "task_expectation": row.get("task_expectation", ""),
            }),
        })

        messages = row.get("messages") or []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = _text(_coalesce(msg, "type", "role", default="")).lower()
            text = _text(_coalesce(msg, "content", "text", default=""))
            event_id = _text(_coalesce(msg, "id", "message_id", default=f"{episode_id}_{idx}"))
            ts = _timestamp(msg.get("timestamp", ""))

            events.append({
                "source": source,
                "user_id": user_id,
                "episode_id": episode_id,
                "event_id": event_id,
                "event_index": idx,
                "timestamp": ts,
                "role": role,
                "text": text,
                "metadata_json": _json({}),
            })

            # Handle both published nested schema variants.
            thought_groups = []
            if "reasons" in msg:
                thought_groups.append(("reason", msg.get("reasons") or []))
            if "reactions" in msg:
                thought_groups.append(("reaction", msg.get("reactions") or []))
            if "thoughts" in msg:
                generic = msg.get("thoughts") or []
                for th in generic:
                    if isinstance(th, dict):
                        thought_groups.append((_text(th.get("thought_type", "thought")), [th]))

            for default_type, group in thought_groups:
                for th in group:
                    if not isinstance(th, dict):
                        continue
                    stype = _text(_coalesce(th, "thought_type", default=default_type)) or default_type
                    stext = _text(_coalesce(th, "content", "text", default=""))
                    slabel = _text(th.get("label", ""))
                    signals.append({
                        "source": source,
                        "user_id": user_id,
                        "episode_id": episode_id,
                        "event_id": event_id,
                        "signal_type": stype,
                        "signal_text": stext,
                        "signal_label": slabel,
                        "signal_value": "",
                        "metadata_json": _json({"timestamp": th.get("timestamp", "")}),
                    })

    return {
        "users": pd.DataFrame(users.values()),
        "episodes": pd.DataFrame(episodes),
        "events": pd.DataFrame(events),
        "signals": pd.DataFrame(signals),
    }


TURN_NAMES = ("first", "second", "third", "fourth")


def _resolve_preferred_response(row: Dict[str, Any], turn: str) -> str:
    pref = _text(row.get(f"{turn}_turn_preferred_response", ""))
    if not pref:
        return ""
    p = pref.strip().lower()
    mapping = {
        "a": "a", "b": "b", "c": "c", "d": "d",
        "response_a": "a", "response_b": "b", "response_c": "c", "response_d": "d",
        "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
    }
    key = mapping.get(p)
    if key:
        resolved = _text(row.get(f"{turn}_turn_response_{key}", ""))
        return resolved or pref
    # In some releases the preferred-response field itself contains the text.
    return pref


def adapt_community_alignment_rows(rows: Iterable[Dict[str, Any]], source: str = "community_alignment") -> Dict[str, pd.DataFrame]:
    users: Dict[str, Dict[str, Any]] = {}
    episodes: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    signals: List[Dict[str, Any]] = []

    for row in rows:
        row = dict(row)
        user_id = _text(row.get("annotator_id", ""))
        episode_id = _text(row.get("conversation_id", ""))
        wave = _text(row.get("wave", ""))
        language = _text(row.get("assigned_lang", ""))

        users.setdefault(user_id, {
            "source": source,
            "user_id": user_id,
            "metadata_json": _json({}),
        })

        episodes.append({
            "source": source,
            "user_id": user_id,
            "episode_id": episode_id,
            "start_time": "",
            "end_time": "",
            "model": "multiple_candidate_models",
            "language": language,
            "topic": "",
            "wave": wave,
            "order_basis": "wave_only_within_wave_unordered",
            "metadata_json": _json({
                "is_pregenerated_first_prompt": row.get("is_pregenerated_first_prompt", None),
                "in_balanced_subset": row.get("in_balanced_subset", None),
                "in_balanced_subset_10": row.get("in_balanced_subset_10", None),
            }),
        })

        event_idx = 0
        for turn in TURN_NAMES:
            prompt = _text(row.get(f"{turn}_turn_prompt", ""))
            if not prompt:
                continue

            user_event_id = f"{episode_id}_{turn}_user"
            events.append({
                "source": source,
                "user_id": user_id,
                "episode_id": episode_id,
                "event_id": user_event_id,
                "event_index": event_idx,
                "timestamp": "",
                "role": "user",
                "text": prompt,
                "metadata_json": _json({"turn": turn}),
            })
            event_idx += 1

            preferred = _resolve_preferred_response(row, turn)
            if preferred:
                assistant_event_id = f"{episode_id}_{turn}_preferred_assistant"
                # This is the human-selected response, not necessarily a single model's live reply.
                events.append({
                    "source": source,
                    "user_id": user_id,
                    "episode_id": episode_id,
                    "event_id": assistant_event_id,
                    "event_index": event_idx,
                    "timestamp": "",
                    "role": "assistant_preferred",
                    "text": preferred,
                    "metadata_json": _json({"turn": turn}),
                })
                event_idx += 1

                feedback = _text(row.get(f"{turn}_turn_feedback", ""))
                # Preserve the full candidate set for preference supervision.
                # Only the human-selected response is placed on the observed
                # conversational path; rejected alternatives live in signal metadata.
                candidates = {
                    key: _text(row.get(f"{turn}_turn_response_{key}", ""))
                    for key in ("a", "b", "c", "d")
                }
                raw_preferred = _text(row.get(f"{turn}_turn_preferred_response", ""))
                signals.append({
                    "source": source,
                    "user_id": user_id,
                    "episode_id": episode_id,
                    "event_id": user_event_id,
                    "signal_type": "response_preference",
                    "signal_text": preferred,
                    "signal_label": raw_preferred,
                    "signal_value": "",
                    "metadata_json": _json({
                        "turn": turn,
                        "feedback": feedback,
                        "candidate_responses": candidates,
                        "raw_preferred_response": raw_preferred,
                    }),
                })

    return {
        "users": pd.DataFrame(users.values()),
        "episodes": pd.DataFrame(episodes),
        "events": pd.DataFrame(events),
        "signals": pd.DataFrame(signals),
    }


def _load_json_any(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        # Common wrappers.
        for key in ("data", "records", "conversations", "items"):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
        # Dict keyed by IDs.
        if all(isinstance(v, dict) for v in obj.values()):
            out = []
            for k, v in obj.items():
                x = dict(v)
                x.setdefault("_record_key", k)
                out.append(x)
            return out
    return []


def adapt_talk2ai_local(raw_dir: Path, source: str = "talk2ai") -> Dict[str, pd.DataFrame]:
    """Adapt a local Talk2AI release.

    Expected public-paper filenames:
      talk2ai_registries.json
      talk2ai_psyscales.json
      plus a conversation-log JSON file.

    The paper documents the first two names but search indexing does not expose
    a stable public download URL for the repository. Therefore this adapter does
    not guess a URL. Put the release files in data/raw/talk2ai/ and it will
    discover the conversation JSON by exclusion / filename heuristics.
    """
    raw_dir = Path(raw_dir)
    registry_path = raw_dir / "talk2ai_registries.json"
    scales_path = raw_dir / "talk2ai_psyscales.json"
    if not registry_path.exists() or not scales_path.exists():
        raise FileNotFoundError(
            "Talk2AI local files not found. Expected at least:\n"
            f"  {registry_path}\n  {scales_path}\n"
            "Place the public Talk2AI JSON release in data/raw/talk2ai/."
        )

    candidates = [p for p in raw_dir.glob("*.json") if p not in {registry_path, scales_path}]
    if not candidates:
        raise FileNotFoundError("Could not find Talk2AI conversation-log JSON in the same folder.")
    # Prefer filenames containing conversation/chat/message.
    candidates.sort(key=lambda p: (not bool(re.search(r"conversation|chat|message", p.name, re.I)), p.name))
    convo_path = candidates[0]

    registries = _iter_records(_load_json_any(registry_path))
    scales = _iter_records(_load_json_any(scales_path))
    convos = _iter_records(_load_json_any(convo_path))

    users: Dict[str, Dict[str, Any]] = {}
    reg_lookup: Dict[str, Dict[str, Any]] = {}
    for r in registries:
        uid = _text(_coalesce(r, "userId", "user_id", "participant_id", "_record_key", default=""))
        if not uid:
            continue
        reg_lookup[uid] = r
        users[uid] = {
            "source": source,
            "user_id": uid,
            "metadata_json": _json({
                # Keep only experimental assignment by default; omit demographics.
                "llm": _coalesce(r, "llms", "llm", "model", default=""),
                "topic": _coalesce(r, "topic", default=""),
            }),
        }

    # Feedback/psychometric records become signals. We do not hard-code exact
    # keys beyond Q0-Q4; preserve the raw record in metadata for EDA.
    signals: List[Dict[str, Any]] = []
    scale_by_user = defaultdict(list)
    for s in scales:
        uid = _text(_coalesce(s, "userId", "user_id", "participant_id", default=""))
        if uid:
            scale_by_user[uid].append(s)

    episodes: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for ci, c in enumerate(convos):
        uid = _text(_coalesce(c, "userId", "user_id", "participant_id", default=""))
        if not uid:
            continue
        episode_id = _text(_coalesce(c, "conversationId", "conversation_id", "sessionId", "session_id", "_record_key", default=f"{uid}_{ci}"))
        wave = _text(_coalesce(c, "wave", "session", "session_number", default=""))
        reg = reg_lookup.get(uid, {})
        model = _text(_coalesce(c, "llm", "model", default=_coalesce(reg, "llms", default="")))
        topic = _text(_coalesce(c, "topic", default=_coalesce(reg, "topic", default="")))
        ts = _timestamp(_coalesce(c, "timestamp", "insertDate", "created_at", default=""))

        episodes.append({
            "source": source,
            "user_id": uid,
            "episode_id": episode_id,
            "start_time": ts,
            "end_time": "",
            "model": model,
            "language": "it",
            "topic": topic,
            "wave": wave,
            "order_basis": "weekly_wave_then_timestamp",
            "metadata_json": _json({}),
        })

        msgs = _coalesce(c, "messages", "conversation", "turns", default=[])
        if isinstance(msgs, dict):
            msgs = list(msgs.values())
        if not isinstance(msgs, list):
            msgs = []

        for idx, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            role = _text(_coalesce(m, "role", "type", "speaker", default="")).lower()
            if role in {"human", "participant"}:
                role = "user"
            elif role in {"ai", "model", "bot"}:
                role = "assistant"
            text = _text(_coalesce(m, "content", "text", "message", default=""))
            eid = _text(_coalesce(m, "id", "message_id", default=f"{episode_id}_{idx}"))
            events.append({
                "source": source,
                "user_id": uid,
                "episode_id": episode_id,
                "event_id": eid,
                "event_index": idx,
                "timestamp": _timestamp(_coalesce(m, "timestamp", default="")),
                "role": role,
                "text": text,
                "metadata_json": _json({}),
            })

    # Attach post-session signals by user; exact episode linkage is dataset-schema dependent.
    for uid, records in scale_by_user.items():
        for si, s in enumerate(records):
            keys = {k.lower(): k for k in s.keys()}
            for q in ("q0", "q1", "q2", "q3", "q4"):
                source_key = next((orig for low, orig in keys.items() if low == q or low.endswith(q)), None)
                if source_key is None:
                    continue
                val = s.get(source_key)
                signals.append({
                    "source": source,
                    "user_id": uid,
                    "episode_id": _text(_coalesce(s, "sessionId", "session_id", "wave", "session", default="")),
                    "event_id": "",
                    "signal_type": f"talk2ai_{q}",
                    "signal_text": _text(val) if q == "q4" else "",
                    "signal_label": q,
                    "signal_value": _text(val) if q != "q4" else "",
                    "metadata_json": _json({"raw_record_index": si}),
                })

    return {
        "users": pd.DataFrame(users.values()),
        "episodes": pd.DataFrame(episodes),
        "events": pd.DataFrame(events),
        "signals": pd.DataFrame(signals),
    }


def write_tables(tables: Dict[str, pd.DataFrame], out_dir: Path, fmt: str = "csv") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_TABLES:
        df = tables.get(name, pd.DataFrame())
        if fmt == "parquet":
            try:
                df.to_parquet(out_dir / f"{name}.parquet", index=False)
                continue
            except Exception as exc:
                print(f"[{name}] parquet unavailable ({exc}); falling back to CSV")
        df.to_csv(out_dir / f"{name}.csv", index=False)


def merge_sources(source_tables: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    merged = {}
    for table in CANONICAL_TABLES:
        frames = [tabs.get(table, pd.DataFrame()) for tabs in source_tables.values()]
        frames = [f for f in frames if not f.empty]
        merged[table] = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return merged
