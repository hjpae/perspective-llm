# ATS 06 — CEAR-style recurrent g: implementation notes

This is the first mainline ATS implementation. It is intentionally **not** a static history compressor.

## Architectural invariant

`g_(t-1) -> organization of z_t -> prediction -> observed human evidence -> g_t`

The current default uses a state-conditioned low-rank SPD metric over the **full frozen semantic embedding space**:

`M_g = D_g + U_g U_g^T`

There is no projection before the metric. `M_g z` is evaluated implicitly in `O(d r)` time. At `g=0`, `M_g=I` exactly, so the reset condition is a clean query-only baseline through the same trained model.

The recurrent update is CEAR-style:

`candidate_t = LN(GRU(organized_evidence_t, g_(t-1)))`

`alpha_t = alpha_bar + alpha_range/2 * tanh(AlphaNet(...))`

`g_t = (1-alpha_t) * g_(t-1) + alpha_t * candidate_t`

`alpha_t` is per-dimension and initialized near 0.08, bounded to `[0.03, 0.30]`.

## Community Alignment limitation

CA has repeated observations per person but no trustworthy across-conversation chronology. Therefore the implementation preserves recurrence but treats order as a nuisance:

- support order is randomly permuted every training epoch;
- validation/test predictions are averaged across multiple support permutations;
- `support_order_stability.csv` measures how strongly inferred `g` depends on arbitrary support order.

This experiment can test whether a compact recurrent person-state helps prediction. It cannot establish real temporal revision dynamics.

## First-run comparisons

The script outputs:

- `g_learned`
- `g_reset_query_only`
- `g_swapped_same_language`

The key desired pattern is lower NLL/Brier for `g_learned` than both reset and swapped state. Accuracy is secondary.

## Important outputs

- `test_summary.csv`
- `bootstrap_deltas.csv`
- `support_order_stability.csv`
- `alpha_diagnostics.csv`
- `test_g_states.csv`
- query-level prediction CSVs
- `training_history.csv`
- `experiment_metadata.json`

## Next ablations after the first successful run

Do not add these before seeing whether the mainline model learns anything:

1. FiLM coupler using the same recurrent updater.
2. Fixed-alpha / vanilla-GRU updater.
3. Static pooled-profile baseline.
4. Strong semantic retrieval/RAG with the same current query.
5. Same RAG + g, followed by the personalization-efficiency frontier.
