# perspective-llm

**Can a slow, history-dependent perspective state be implemented outside a frozen LLM, and does it actually change how the model interprets new evidence?**

This repository contains a proof-of-mechanism study transferring the perspective architecture developed in [CEAR Lab](https://docs.superhuman.com/d/AII_do5M5iYau8H/Perspective-Architectures-for-Coherent-and-Attuned-Artificial-Ag_suaEkYpn#_lurWkl7-) — a slow latent state, decoupled from immediate task objectives, whose accumulated history conditions how new observations are interpreted — from minimal simulated agents into an LLM-based persistent-agent setting.

The central design commitment is a separation of responsibilities:

| Component | Responsibility |
|---|---|
| **LLM (frozen, API)** | semantic parsing and interpretation |
| **Explicit Python updater** | state dynamics, revision rate, hysteresis |
| **Persistent state `g`** | what the system believes about the person, with evidence and provenance |

The state is **never rewritten directly by the response model**. The LLM reads it and interprets against it; a deterministic updater decides whether, and how much, it changes. This separation is what makes the state inspectable and its dynamics auditable.

> **Scope.** These are toy-scale, single-scenario experiments establishing *engineering feasibility*, not scientific superiority. No comparison against strong baselines (RAG, LLM-maintained structured profiles) has been run yet. Update-law constants are explicit design heuristics, not calibrated parameters. See [Limitations](#limitations).

---

## Architecture

```
utterance x_t
    ↓
LLM semantic parser              → raw semantic observation s_t
    ↓
deterministic relation resolver  + g_(t-1)
    ↓
history-conditioned interpretation z_t
    ↓
explicit hysteretic updater      → revision rate α_t
    ↓
persistent state g_t
```

The persistent state distinguishes three layers, each carrying its own evidence:

- **`established_state`** — durable beliefs about the person, with `confidence`, `stability`, and `last_confirmed_step`.
- **`change_tracker`** — accumulated evidence for a candidate direction of change, with `score`, `qualifying_evidence_count`, activation/deactivation history, and `provenance_ids` for every contributing observation.
- **`provisional_change`** — an active hypothesis that the established state may be changing, held separately from the established state itself rather than overwriting it.

Separating *provisional* from *established* is what allows the system to register a possible change without committing to it — and to keep a record of having considered it after the evidence recedes.

---

## Experiments and results

### 01 — Does a given state condition interpretation?

Same present evidence `x_t`, different hand-constructed slow state `g_(t-1)` → different interpretation `z_t` and different revision openness `α_t`.

**Result: yes.** Establishes the causal pathway before asking whether history can build the state on its own.

### 02 — Can history itself form the state?

Two longitudinal histories from an identical initial state: (A) stable baseline → one anomaly → return to baseline; (B) stable baseline → repeated contradictory evidence. Both then receive the **same** final probe: *"I don't feel like going outside today."*

| | after transient history | after sustained-change history |
|---|---|---|
| `relation_to_provisional_change` | `no_active_change` | `supports` |
| `evidence_for_longitudinal_change` | 0.30 | 0.60 |
| `confidence` | 0.55 | 0.65 |

**Result: yes.** Identical input, divergent reading — the difference is carried entirely by accumulated history.

### 03 — Does the state resist immediate reversal? (hysteresis)

Three scenarios sharing the same established state, differing only in evidence trajectory.

| scenario | activation step | deactivation step | final status | final score | qualifying evidence | peak score |
|---|---|---|---|---|---|---|
| `transient_return` | — (never) | — | `inactive` | 0.000 | 1 | 0.228 |
| `change_then_recovery` | 9 | 18 | `monitoring` | 0.419 | 6 | 1.535 |
| `persistent_change` | 10 | — | `active` | 1.040 | 6 | 1.424 |

Two things matter here.

**Asymmetric thresholds.** Activation at step 9 and deactivation at step 18 are governed by different thresholds, so the state's path through evidence — not just its present evidence — determines its condition. A single transient signal never crosses activation at all (peak 0.228).

**Residue after recovery.** `transient_return` and `change_then_recovery` both end without an active change, but they are not in the same internal condition: the first ends `inactive` with score 0.0 and one qualifying observation; the second ends `monitoring` with score 0.419, six qualifying observations, and a preserved peak of 1.535. **A system that has been through a change is not identical to one that never was, even when both currently register no change.**

That residue has behavioral consequences. Probing both with the identical utterance *"I took my usual morning walk and enjoyed it"*:

| | after transient history | after sustained-change history |
|---|---|---|
| `candidate_change_direction` | `none` | `increased_outdoor_engagement` |
| `relation_to_provisional_change` | `no_active_change` | `contradicts` |
| `evidence_for_longitudinal_change` | 0.20 | 0.60 |
| **revision rate `α_t`** | **0.0167** | **0.2295** |

The same sentence is read as unremarkable confirmation of a stable habit in one history, and as counter-evidence against an active provisional change in the other — and it moves the state **13.7× more** in the second case. This is the LLM-setting analogue of the latent-swap result from the minimal-agent work.

### 04 — Long-horizon stress test (200 synthetic days)

A single continuous timeline: stable baseline (1–60) → transient perturbation (61–65) → restabilization (66–100) → genuine sustained change (101–145) → mixed/conflicting evidence (146–160) → sustained recovery (161–200).

| metric | value |
|---|---|
| false activations before genuine change | **0** |
| first activation step | 105 (change begins at 101) |
| activation latency | **5 steps** |
| first deactivation step | 165 (recovery begins at 161) |
| recovery latency | **5 steps** |
| status switches during mixed-evidence phase | **0** |
| semantic accuracy (overall) | 0.995 |
| semantic accuracy by phase | 1.00 / 0.80 / 1.00 / 1.00 / 1.00 / 1.00 |
| peak change score | 2.0 |
| end state active | false |

This is the two-sided failure test. **Over-updating**: sixty steps of stable baseline plus a five-step perturbation produced zero false activations. **Under-updating**: genuine sustained change was registered within five steps, and recovery within five steps of the evidence turning. Under deliberately mixed evidence the tracker did not oscillate even once — the failure mode where a system thrashes between states under ambiguity did not appear.

Note also that the single phase with degraded semantic accuracy (0.80, transient perturbation) did **not** propagate into a state error: parsing failures stay localized because interpretation and state dynamics are separate components.

### 05 — Replication evaluation

Same 200-step scenario across *N* independent LLM runs, with scenario, architecture, relation resolver, hysteresis law, and thresholds all held fixed, isolating stochastic variation in the semantic parser. Produces per-run traces, median trajectories with bootstrap 95% CIs, fraction-active trajectories, and semantic-error-rate trajectories. *(Design implemented; results pending.)*

---

## What this does and does not establish

**Established.** The CEAR slow/history-dependent perspective principle can be implemented on top of a frozen, API-accessed LLM using an external explicit state scaffold. History-conditioned interpretation, asymmetric hysteresis, post-recovery residue, and bounded revision all reproduce in this setting. Engineering feasibility is no longer hypothetical.

**Not established.** Whether this outperforms simpler approaches. The necessary comparisons — against retrieval-augmented generation and against an LLM-maintained evolving structured profile — have not been run. Nor has identifiability of the latent state been tested under realistic, noisy, sparse human interaction histories, which is the sharpest open risk: real users generate contradictory and context-dependent evidence, and the underlying state may itself be changing.

## Limitations

- Toy-scale synthetic scenarios, single domain, single base model.
- Update-law constants (decay rates, activation and deactivation thresholds, α scaling) are explicit design choices, not calibrated from data.
- No strong-baseline comparison; no causal intervention study swapping states across histories while holding evidence and model fixed.
- Semantic parsing depends on a commercial API and inherits its stochasticity; experiment 05 is designed to quantify this but has not yet been run at scale.

## Roadmap

1. Strong baselines: full-context/RAG and an evolving structured user profile, on identical scenarios with identical base models.
2. Causal interventions: hold the current utterance, retrieved evidence, and LLM constant while manipulating or swapping the state across histories.
3. Robustness: multiple models, varied change rates, adversarial and noisy evidence.
4. Metrics: stability, appropriate plasticity, provenance quality, corrigibility.
5. Release: evaluation suite and reusable state layer.

---

## Related work

- **[Pae (2026)](https://www.youtube.com/watch?v=IJUhrqP6y2I)**, *Minimal Computational Preconditions for Subjective Perspective in Artificial Agents* - AAAI Spring Symposium on Machine Consciousness. Original slow perspective-latent architecture in minimal agents.
- **[CEAR Lab, Phase 2](https://github.com/hjpae/cearlab-phase2)** - history-dependent perspective dynamics, stability/plasticity, and persistent effects of perturbation. The direct technical precursor to this transfer.

---

## Setup

### Instance setup (change username if you are not me)

Generate the SSH key:  

```bash
git config --global user.name "hjpae"
git config --global user.email "hnjpae@gmail.com"

ssh-keygen -t ed25519 -C "hnjpae@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add SSH key and then authorize connection to GitHub:  

```bash
ssh -T git@github.com
```

Clone the repository:  

```bash
git clone git@github.com:hjpae/perspective-llm.git
cd perspective-llm
git remote -v
```

If overwriting the existing repo, use this:  

```bash
git remote remove origin
git remote add origin git@github.com:hjpae/perspective-llm.git
```

### Environment

```bash
conda env create -f environment.yml
conda activate perspective-llm
```

Install PyTorch separately if needed:

```bash
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Create a `.env` file in the repository root:

```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-nano
```

Verify the connection:

```bash
python experiments/00_api_smoke_test.py
```

### Running

Experiments are written as cell-delimited scripts (`#%%`) and run top to bottom, each writing traces, final states, and figures to `outputs/<experiment>/`.  
This initial sanity check runs in Spyder IDE (python 3.10, Windows machine) 

```bash
python experiments/02_history_to_state.py
python experiments/03_recovery_hysteresis.py
python experiments/04_long_horizon.py
```

### Repository layout

```
experiments/   numbered experiment scripts, 00–05
outputs/       per-experiment traces (.jsonl), final states (.json), figures (.png)
```
