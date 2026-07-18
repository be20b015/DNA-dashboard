"""
Genomic Dashboard — main Streamlit entry point.

Run with:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

# Cross-platform memory footprint tracking
try:
    import resource
except ImportError:
    import psutil
    resource = None

# Allow `core` / `components` package imports when run as `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ngl_viewer import render_structure
from components.seqviz import render_seqviz
from core.kpis import RunningKPIs
from core.parser import stream_with_progress

st.set_page_config(page_title="Genomic Dashboard", layout="wide", page_icon="🧬")

# ---------------------------------------------------------------------------
# Sidebar — file upload & options
# ---------------------------------------------------------------------------
st.sidebar.title("🧬 Genomic Dashboard")
uploaded_file = st.sidebar.file_uploader(
    "Upload FASTA or FASTQ file",
    type=["fasta", "fa", "fastq", "fq", "txt"],
    help="Files are parsed in-memory (streamed) — nothing is written to disk.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("AI Model")
enable_ai = st.sidebar.checkbox(
    "Enable Genomic Foundation Model embeddings",
    value=False,
    help="Loads a Hugging Face model (e.g. DNABERT-2). Requires internet "
         "access and a GPU is strongly recommended for large batches.",
)
provider = st.sidebar.selectbox("AI Insights provider", ["gemini", "claude"], index=0)

st.sidebar.markdown("---")
uploaded_pdb = st.sidebar.file_uploader(
    "Optional: 3D structure (.pdb)",
    type=["pdb"],
    help="e.g. AlphaFold / ESMFold prediction output",
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Genomic Data Dashboard")

if uploaded_file is None:
    st.info("Upload a FASTA or FASTQ file in the sidebar to get started.")
    st.stop()

raw_bytes = uploaded_file.getvalue()  # buffered in RAM by Streamlit, no disk write
size_mb = len(raw_bytes) / (1024 * 1024)
st.caption(f"Loaded **{uploaded_file.name}** ({size_mb:.1f} MB) — streaming parse in progress…")

progress_bar = st.progress(0.0)
status_text = st.empty()

running = RunningKPIs()
first_record = None
records_for_viewer = []

def _on_progress(n_records, pct):
    progress_bar.progress(pct)
    status_text.text(f"Parsed {n_records:,} records…")

# --- Performance Metric Tracking ---
start_time = time.perf_counter()
peak_memory_kb = 0.0

try:
    for record in stream_with_progress(raw_bytes, progress_callback=_on_progress):
        running.update(record)
        if first_record is None:
            first_record = record
        if len(records_for_viewer) < 200:  # cap what we keep for the interactive viewer
            records_for_viewer.append(record)
        
        # Sample memory footprint periodically during iteration
        if resource:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Adjust for OS variation: macOS measures in bytes, Linux in kilobytes
            if sys.platform == "darwin":
                usage = usage / 1024
            if usage > peak_memory_kb:
                peak_memory_kb = usage
        else:
            # Fallback wrapper using psutil for Windows systems
            usage_bytes = psutil.Process().memory_info().rss
            usage_kb = usage_bytes / 1024
            if usage_kb > peak_memory_kb:
                peak_memory_kb = usage_kb

except ValueError as e:
    st.error(str(e))
    st.stop()

end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Boundary catch to prevent DivisionByZero errors on microscopic files
elapsed_time_adj = max(elapsed_time, 0.001)
records_per_sec = running.n_records / elapsed_time_adj
bases_per_sec = running.total_length / elapsed_time_adj
peak_memory_mb = peak_memory_kb / 1024

progress_bar.empty()
status_text.empty()
st.success(f"Parsed {running.n_records:,} records ({running.total_length:,} total bases).")

# Telemetry Output Container
st.info(
    f"⏱️ **Throughput:** {records_per_sec:,.0f} records/sec ({bases_per_sec:,.0f} bases/sec) "
    f"completed in {elapsed_time:.3f}s | "
    f"🧠 **Peak Memory Footprint:** {peak_memory_mb:.2f} MB"
)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
summary = running.summary_dict()
kpi_cols = st.columns(6)
kpi_cols[0].metric("Records", f"{summary['records']:,}")
kpi_cols[1].metric("Total bases", f"{summary['total_bases']:,}")
kpi_cols[2].metric("Mean length", summary["mean_length"])
kpi_cols[3].metric("GC content", f"{summary['overall_gc_pct']}%")
kpi_cols[4].metric("Min / Max length", f"{summary['min_length']} / {summary['max_length']}")
kpi_cols[5].metric("Ambiguous bases", f"{summary['ambiguous_base_pct']}%")

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
st.subheader("Distributions")
dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    len_df = pd.DataFrame({"length": running.length_samples})
    fig = px.histogram(len_df, x="length", nbins=40, title="Sequence length distribution")
    st.plotly_chart(fig, use_container_width=True)

with dist_col2:
    gc_df = pd.DataFrame({"gc_content": running.gc_samples})
    fig2 = px.histogram(gc_df, x="gc_content", nbins=40, title="Per-record GC content (%)")
    st.plotly_chart(fig2, use_container_width=True)

if len(running.length_samples) < running.n_records:
    st.caption(
        f"Distribution plots are based on a {len(running.length_samples):,}-record "
        f"sample (of {running.n_records:,} total) to keep memory bounded."
    )

# ---------------------------------------------------------------------------
# Sequence viewer
# ---------------------------------------------------------------------------
st.subheader("Sequence Viewer")
if records_for_viewer:
    viewer_ids = [r.id for r in records_for_viewer]
    selected_id = st.selectbox("Select a record to view", viewer_ids)
    selected_record = next(r for r in records_for_viewer if r.id == selected_id)
    layout = st.radio("Layout", ["linear", "circular"], horizontal=True)
    render_seqviz(selected_record.sequence, name=selected_record.id, viewer_type=layout)

# ---------------------------------------------------------------------------
# 3D structure viewer
# ---------------------------------------------------------------------------
if uploaded_pdb is not None:
    st.subheader("3D Protein Structure")
    pdb_text = uploaded_pdb.getvalue().decode("utf-8", errors="ignore")
    render_structure(pdb_text)

# ---------------------------------------------------------------------------
# AI-driven embeddings (optional, heavy)
# ---------------------------------------------------------------------------
if enable_ai:
    st.subheader("Foundation Model Embeddings")
    n_to_embed = st.slider("Number of sequences to embed", 1, min(50, len(records_for_viewer) or 1), min(8, len(records_for_viewer) or 1))
    if st.button("Run embedding extraction"):
        with st.spinner("Loading model and extracting embeddings (first load may take a while)…"):
            try:
                from core.model import embed_sequences, load_gfm, gpu_tokenize_hint

                bundle = load_gfm()
                seqs = [r.sequence for r in records_for_viewer[:n_to_embed]]
                vectors = embed_sequences(seqs, bundle)
                st.write(f"Embedding matrix shape: {vectors.shape}")
                st.dataframe(pd.DataFrame(vectors[:, :10]).round(4), use_container_width=True)
                st.caption(gpu_tokenize_hint())
            except Exception as e:
                st.error(
                    "Could not load the foundation model. This usually means "
                    "there's no internet access to huggingface.co in this "
                    "environment, or `transformers`/`torch` aren't installed.\n\nDetails: {e}"
                )

# ---------------------------------------------------------------------------
# AI Insights (Gemini / Claude summary of KPIs)
# ---------------------------------------------------------------------------
st.subheader("AI Insights")
if st.button("Generate plain-language summary"):
    with st.spinner(f"Asking {provider} for an interpretation…"):
        try:
            from core.ai_insights import get_insights

            text = get_insights(summary, provider=provider)
            st.markdown(text)
        except RuntimeError as e:
            st.warning(
                f"{e}. Set the relevant API key as an environment variable "
                "(GEMINI_API_KEY or ANTHROPIC_API_KEY) to enable this feature."
            )
        except Exception as e:
            st.error(f"AI insights request failed: {e}")
