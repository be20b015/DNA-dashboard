# Genomic Dashboard

A high-performance, extensible dashboard for exploring gigabyte-scale
genomic data, with optional foundation-model (DNABERT-2 / HyenaDNA)
embeddings and LLM-generated plain-language insights.

## Features

- **In-memory streaming FASTA/FASTQ parser** — handles multi-GB files
  without writing to disk, constant-ish memory footprint via generators.
- **Live KPIs** — record count, total bases, GC content, length
  distribution, ambiguous-base rate — computed with running counters,
  not full in-memory storage.
- **Interactive sequence viewer** — linear/circular DNA view (via
  seqviz.js).
- **3D structure viewer** — cartoon rendering of uploaded PDB files
  (via NGL.js), for AlphaFold/ESMFold output.
- **Genomic Foundation Model embeddings** — optional DNABERT-2
  integration via Hugging Face `transformers`, cached as a singleton
  so weights load once per server process.
- **AI Insights** — sends KPI summaries to Gemini or Claude for a
  plain-language biological interpretation.
- **FastAPI backend** — decouples heavy inference from the UI process,
  with a minimal role-based access control example and audit logging
  scaffold (a starting point toward 21 CFR Part 11-style logging, not
  a compliance guarantee).

## Project layout

```
genomic_dashboard/
├── app.py                  # Streamlit front end (run this)
├── requirements.txt
├── .streamlit/config.toml  # theme + 4GB max upload size
├── core/
│   ├── parser.py            # streaming FASTA/FASTQ parsing
│   ├── kpis.py               # running KPI aggregation
│   ├── model.py               # GFM loading + embedding extraction
│   └── ai_insights.py          # Gemini/Claude KPI summarization
├── components/
│   ├── seqviz.py             # linear/circular sequence viewer (CDN JS)
│   └── ngl_viewer.py          # 3D structure viewer (CDN JS)
└── backend/
    └── main.py               # FastAPI service (optional, separate process)
```

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Open the local URL Streamlit prints (defaults to `http://localhost:8501`).

Upload any `.fasta`/`.fastq` file in the sidebar — KPIs, distribution
charts, and the sequence viewer populate automatically.

### Enabling foundation-model embeddings

This step downloads model weights (~117M params for DNABERT-2) from
Hugging Face Hub the first time it runs, so it needs outbound internet
access. Toggle "Enable Genomic Foundation Model embeddings" in the
sidebar, then click "Run embedding extraction". A GPU is recommended
for anything beyond small batches — CPU inference works but is slow.

If you're behind a restricted network, pre-download the model on a
machine with internet access and point `TRANSFORMERS_CACHE` /
`HF_HOME` at the same cache directory, or set `HF_HUB_OFFLINE=1` once
cached.

### Enabling AI Insights

Set one of these environment variables before launching:

```bash
export GEMINI_API_KEY=your-key      # for the Gemini provider
export ANTHROPIC_API_KEY=your-key   # for the Claude provider
```

Pick the provider in the sidebar, then click "Generate plain-language
summary" under AI Insights.

### Running the FastAPI backend separately (optional)

For multi-user deployments, run inference as its own service so the
Streamlit process(es) stay lightweight:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Demo API keys are hardcoded for illustration only
(`demo-analyst-key`, `demo-admin-key` in `backend/main.py`) — replace
`API_KEY_ROLES` with a real identity provider (OIDC/JWT, your
institution's SSO, etc.) before handling real patient data. Same goes
for the audit logger, which currently writes to stdout/log file; swap
in an append-only store for anything regulatory.

## Notes on the original spec

A couple of adjustments from the initial prompt list, made during
implementation:

- **`st-seqviz`** isn't a real, maintained PyPI package as far as I
  could verify, so the sequence viewer embeds the `seqviz` JavaScript
  library directly via CDN inside a Streamlit HTML component instead.
  Same approach for the NGL 3D viewer.
- **`DNAtok`** (GPU-accelerated tokenizer) is mentioned in the UI as a
  recommendation rather than wired in automatically, since it's an
  extra dependency you'd need to opt into deliberately.
- The FastAPI RBAC/audit-log code is a *pattern*, not a compliance
  certification — treat the 21 CFR Part 11 mention in the original
  spec as "build toward this," not "this satisfies it out of the box."

## Scaling tips

- Use `st.cache_data` for DataFrame-shaped results, `st.cache_resource`
  for shared objects like loaded models (both already used here).
- On Hugging Face Spaces or similar ephemeral-disk platforms, set
  `HF_HOME` / `TRANSFORMERS_CACHE` to a persistent volume so model
  weights don't re-download on every restart.
- For simple classification tasks, embedding + a lightweight
  classifier (e.g. logistic regression / small MLP on top of pooled
  embeddings) is typically 10-20x faster than fine-tuning the full
  foundation model, and cheaper to serve.
