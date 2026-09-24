# ATS real-human corpus pipeline

## Directory layout

```text
perspective-llm/
├─ ATS/
│  ├─ data_pipeline/
│  │  ├─ real_corpora.py
│  │  └─ ats_02_prepare_real_corpora.py
│  ├─ eda/
│  │  └─ ats_03_real_corpora_eda.py
│  └─ README_REAL_CORPORA.md
├─ data/
│  ├─ raw/
│  │  └─ talk2ai/                  # add Talk2AI JSON release here later
│  └─ ATS/
│     └─ canonical/
│        ├─ thoughttrace/
│        ├─ community_alignment/
│        ├─ talk2ai/
│        └─ combined/
└─ results/
   └─ ATS/
      └─ eda/
         └─ real_corpora/
```

## Why four canonical tables?

The shared schema keeps corpus-specific labels out of the learned state.

- `events`: what happened in the human–AI interaction. This is the default ATS input.
- `signals`: human preference, thought, reaction, or psychometric outcome. Use as supervision/probes/evaluation, not as state ontology.
- `episodes`: session/conversation context and ordering information.
- `users`: stable source-local user IDs plus deliberately minimal metadata.

## First run

From the repository-root conda environment:

```bash
pip install datasets pandas numpy matplotlib huggingface_hub
```

In Spyder, make the repository root the working directory, open:

`ATS/data_pipeline/ats_02_prepare_real_corpora.py`

and Run File.

The defaults:

- load all ThoughtTrace (~2.2k conversations)
- stream only the first 5,000 Community Alignment conversations for EDA
- leave Talk2AI off until its public JSON files are placed in `data/raw/talk2ai/`

Then run:

`ATS/eda/ats_03_real_corpora_eda.py`

## Corpus roles (working hypothesis, not hard-coded)

- ThoughtTrace: user-side reasons/reactions; useful for latent-state probes and next-behavior prediction.
- Community Alignment: many repeated conversations per annotator; useful for learning persistent user-specific preference state at scale.
- Talk2AI: four weekly sessions with repeated human measurements; useful for true longitudinal validation once local files are available.
- PersonaMem: keep separately as a controlled synthetic intervention benchmark, not the core state-learning corpus.

## Important modeling rule

Do not concatenate `signals` into the ATS input by default.
The core learned state should be formed from interaction `events` and evaluated against signals afterward.
