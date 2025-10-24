
# app.py
import io
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import chardet

try:
    from polyfuzz import PolyFuzz
except Exception as e:
    PolyFuzz = None

st.set_page_config(page_title="Keyword Clustering (PolyFuzz)", page_icon="🔎", layout="wide")

# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class IntentConfig:
    info_pattern: str = (r"\bwhat|where|why|when|who|how|which|guide|tutorial|tips|learn|"
                         r"definition|meaning|best way|ideas|examples|vs\b")
    nav_pattern: str = (r"\blogin|homepage|dashboard|pricing|contact|about|careers|"
                        r"brand|official site|docs\b")
    trans_pattern: str = (r"\bbuy|order|price|cost|deal|discount|sale|coupon|book|"
                          r"download|trial|demo|quote|subscribe|signup|get started\b")

@dataclass
class ClusterConfig:
    model: str = "TF-IDF"
    min_similarity: float = 0.75   # interactive view
    batch_thresholds: Tuple[float, ...] = (0.6, 0.7, 0.8)  # multi-batch
    parent_strategy: str = "Highest metric"  # or Cluster representative/Longest/Shortest
    parent_metric_col: Optional[str] = None  # used when parent_strategy == "Highest metric"
    keyword_col_candidates: Tuple[str, ...] = ("Keyword", "Query", "Search term")
    url_col_candidates: Tuple[str, ...] = ("URL", "Page", "Page URL", "Landing Page", "Final URL")
    impressions_col_candidates: Tuple[str, ...] = ("Impressions", "Clicks", "Sessions", "Pageviews")
    volume_col_candidates: Tuple[str, ...] = ("Search Volume", "Volume", "SV")
    intent: IntentConfig = field(default_factory=IntentConfig)
    cluster_url_strategy: str = "From parent row"  # or Most frequent URL / Highest metric URL
    cluster_separately_by_url: bool = False

# -----------------------------
# Utils
# -----------------------------
def detect_encoding_from_bytes(b: bytes) -> str:
    res = chardet.detect(b)
    return res.get("encoding") or "utf-8"

def read_table_from_upload(f) -> pd.DataFrame:
    name = f.name.lower()
    content = f.read()
    if name.endswith(".csv") or name.endswith(".tsv"):
        encoding = detect_encoding_from_bytes(content)
        sep = "," if name.endswith(".csv") else "\t"
        return pd.read_csv(io.BytesIO(content), encoding=encoding, sep=sep)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file type. Please upload CSV, TSV, or XLSX.")

def _find_first(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        lc = cand.lower()
        if lc in cols:
            return cols[lc]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def detect_columns(df: pd.DataFrame, cfg: ClusterConfig) -> Dict[str, Optional[str]]:
    key_col = _find_first(df, cfg.keyword_col_candidates)
    url_col = _find_first(df, cfg.url_col_candidates)
    imp_col = _find_first(df, cfg.impressions_col_candidates)
    vol_col = _find_first(df, cfg.volume_col_candidates)
    return {"keyword": key_col, "url": url_col, "impressions": imp_col, "volume": vol_col}

def normalize_keywords(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
              .str.lower())

@st.cache_data(show_spinner=False)
def cluster_keywords_cached(keywords: List[str], model_name: str) -> pd.DataFrame:
    if PolyFuzz is None:
        raise ImportError("polyfuzz is required. Install with: pip install 'polyfuzz[fast]'")
    model = PolyFuzz(model_name)
    model.match(keywords, keywords)
    matches = model.get_matches()
    df = matches.rename(columns={"From": "keyword", "To": "cluster_repr", "Similarity": "similarity"})
    return df

def choose_parent(df_group: pd.DataFrame, strategy: str, metric_col: Optional[str]) -> str:
    if strategy == "Highest metric" and metric_col in df_group.columns:
        # Pick the keyword with the highest metric
        ser = pd.to_numeric(df_group[metric_col], errors="coerce").fillna(0)
        return df_group.loc[ser.idxmax(), "keyword"]
    elif strategy == "Longest keyword":
        return df_group.loc[df_group["keyword"].str.len().idxmax(), "keyword"]
    elif strategy == "Shortest keyword":
        return df_group.loc[df_group["keyword"].str.len().idxmin(), "keyword"]
    else:
        # Cluster representative (default/fallback)
        # Use the most common cluster_repr string within group
        return Counter(df_group["cluster_repr"]).most_common(1)[0][0]

def assign_parent_names(df: pd.DataFrame, cols: Dict[str, Optional[str]], cfg: ClusterConfig) -> pd.DataFrame:
    metric_col = cfg.parent_metric_col
    if cfg.parent_strategy == "Highest metric" and not metric_col:
        # auto-pick metric col (prefer Impressions, else Volume, else first numeric col)
        for c in [cols.get("impressions"), cols.get("volume")]:
            if c and c in df.columns:
                metric_col = c
                break
        if not metric_col:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["similarity"]]
            metric_col = numeric_cols[0] if numeric_cols else None

    # Compute parent per cluster_repr
    parents = []
    for cluster_id, sub in df.groupby("cluster_repr", sort=False):
        parent = choose_parent(sub, cfg.parent_strategy, metric_col)
        parents.append((cluster_id, parent))
    parent_map = dict(parents)
    df["parent"] = df["cluster_repr"].map(parent_map)
    return df

def add_intents(df: pd.DataFrame, cfg: IntentConfig, keyword_col: str = "keyword") -> pd.DataFrame:
    intent = pd.Series(index=df.index, dtype="string")
    kw = df[keyword_col].astype(str)
    info_mask = kw.str.contains(cfg.info_pattern, case=False, regex=True, na=False)
    nav_mask = kw.str.contains(cfg.nav_pattern, case=False, regex=True, na=False)
    trans_mask = kw.str.contains(cfg.trans_pattern, case=False, regex=True, na=False)
    intent[info_mask] = "Informational"
    intent[nav_mask] = "Navigational"
    intent[trans_mask] = "Transactional"
    intent = intent.fillna("Unmapped")
    df["intent"] = intent
    return df

def assign_cluster_url(df: pd.DataFrame, url_col: Optional[str], cfg: ClusterConfig) -> pd.DataFrame:
    if not url_col or url_col not in df.columns:
        return df  # nothing to do

    cluster_urls = {}
    for cluster_id, sub in df.groupby("parent", sort=False):
        if cfg.cluster_url_strategy == "From parent row":
            # find the row where keyword == parent
            parent_row = sub.loc[sub["keyword"] == cluster_id]
            if parent_row.empty:
                parent_row = sub.loc[sub["keyword"] == sub["parent"]]
            if parent_row.empty:
                # fallback to most frequent
                url = sub[url_col].mode().iloc[0] if not sub[url_col].isna().all() else None
            else:
                url = parent_row[url_col].iloc[0]
        elif cfg.cluster_url_strategy == "Most frequent URL":
            url = sub[url_col].mode().iloc[0] if not sub[url_col].isna().all() else None
        else:  # Highest metric URL
            # choose by metric if available
            metric_col = cfg.parent_metric_col
            if not metric_col or metric_col not in sub.columns:
                metric_col = None
                candidates = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c]) and c not in ["similarity"]]
                if candidates:
                    metric_col = candidates[0]
            if metric_col:
                ser = pd.to_numeric(sub[metric_col], errors="coerce").fillna(0)
                url = sub.loc[ser.idxmax(), url_col]
            else:
                url = sub[url_col].mode().iloc[0] if not sub[url_col].isna().all() else None
        cluster_urls[cluster_id] = url

    df["cluster_url"] = df["parent"].map(cluster_urls)
    return df

def run_pipeline(df_in: pd.DataFrame, cfg: ClusterConfig, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    # Prepare base
    df = df_in.copy()
    df.rename(columns={cols["keyword"]: "Keyword"}, inplace=True)
    df["keyword"] = normalize_keywords(df["Keyword"])

    # Cluster once and reuse for batches
    clustered = cluster_keywords_cached(df["keyword"].tolist(), cfg.model)
    merged = clustered.merge(df, on="keyword", how="left")

    results = {}
    thresholds = sorted(set([round(float(t), 2) for t in cfg.batch_thresholds if 0 <= float(t) <= 1]))
    for thr in thresholds:
        temp = merged[merged["similarity"] >= thr].copy()
        temp = assign_parent_names(temp, cols, cfg)
        temp = add_intents(temp, cfg.intent, keyword_col="keyword")
        temp = assign_cluster_url(temp, cols.get("url"), cfg)

        # Reorder columns
        front_cols = ["keyword", "parent", "cluster_repr", "similarity", "intent", "cluster_url"]
        # Ensure URL is present visibly
        if cols.get("url") and cols["url"] in temp.columns and cols["url"] not in front_cols:
            front_cols.append(cols["url"])
        other_cols = [c for c in temp.columns if c not in front_cols]
        final = temp[front_cols + other_cols].reset_index(drop=True)
        results[thr] = final
    return results

def run_pipeline_grouped_by_url(df_in: pd.DataFrame, cfg: ClusterConfig, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    url_col = cols.get("url")
    if not url_col:
        return run_pipeline(df_in, cfg, cols)

    # Process each URL bucket separately and concat
    outputs = {}
    thresholds = sorted(set([round(float(t), 2) for t in cfg.batch_thresholds if 0 <= float(t) <= 1]))
    for thr in thresholds:
        parts = []
        for url_val, sub in df_in.groupby(url_col, dropna=False):
            cols_sub = cols.copy()
            # sub-detect key column name still the same
            res_thr = run_pipeline(sub, ClusterConfig(**{**cfg.__dict__, "batch_thresholds": (thr,)}), cols_sub)
            parts.append(res_thr[thr])
        outputs[thr] = pd.concat(parts, ignore_index=True)
    return outputs

# -----------------------------
# UI
# -----------------------------
st.title("🔎 Keyword Clustering (PolyFuzz)")
st.caption("Upload a CSV/XLSX containing a keyword column (e.g., *Keyword*, *Query*, or *Search term*). Optionally include a URL column to map keywords → URLs.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Matching model", ["TF-IDF", "EditDistance"], index=0)
    min_sim = st.slider("Interactive min similarity", 0.0, 1.0, 0.75, 0.01)

    batch_str = st.text_input("Batch similarity thresholds (comma-separated)", "0.6,0.7,0.8")
    try:
        batch_thresholds = tuple(sorted(set(float(x.strip()) for x in batch_str.split(",") if x.strip() != "")))
    except Exception:
        batch_thresholds = (0.6, 0.7, 0.8)
        st.warning("Could not parse thresholds. Using 0.6, 0.7, 0.8.")

    st.markdown("---")
    st.subheader("Parent naming")
    parent_strategy = st.selectbox("Strategy", ["Highest metric", "Cluster representative", "Longest keyword", "Shortest keyword"], index=0)
    parent_metric_col_manual = st.text_input("Metric column (optional, auto-detected if blank)", "")

    st.markdown("---")
    st.subheader("URL mapping")
    cluster_url_strategy = st.selectbox("Cluster URL strategy", ["From parent row", "Most frequent URL", "Highest metric URL"], index=0)
    cluster_separately_by_url = st.toggle("Cluster separately per URL (bucket by URL first)", value=False)

    st.markdown("---")
    st.subheader("Intent Regex (optional)")
    info_pat = st.text_area("Informational pattern", IntentConfig().info_pattern, height=80)
    nav_pat = st.text_area("Navigational pattern", IntentConfig().nav_pattern, height=60)
    trans_pat = st.text_area("Transactional pattern", IntentConfig().trans_pattern, height=60)

    st.markdown("---")
    st.caption("Tip: Multi-batch lets you compare different similarity cuts side-by-side.")

uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "tsv", "xlsx", "xls"])

example = st.toggle("Use example data (overrides upload)", value=False)
if example:
    uploaded = None
    df_in = pd.DataFrame({
        "Keyword": ["best running shoes", "buy running shoes", "what are trail shoes",
                    "nike pegasus review", "asics gel-kayano price", "trail runners vs hiking shoes",
                    "adidas ultraboost sale", "running shoe size guide", "brooks ghost 15", "shoe store near me"],
        "Impressions": [1000, 800, 600, 500, 450, 400, 700, 300, 350, 900],
        "URL": [" /shoes/best " , "/shoes/buy", "/shoes/trail", "/shoes/pegasus", "/shoes/gel-kayano",
                "/shoes/trail", "/shoes/ultraboost", "/guides/sizing", "/shoes/brooks-ghost", "/stores/near-me"]
    })
    st.info("Using built-in example dataset.")
elif uploaded is not None:
    try:
        df_in = read_table_from_upload(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    df_in = None

if df_in is not None:
    st.subheader("Preview input")
    st.dataframe(df_in.head(20), use_container_width=True)

    # Detect columns
    cfg_probe = ClusterConfig()
    cols = detect_columns(df_in, cfg_probe)

    # Let user override detected URL or keyword column if needed
    with st.expander("Column detection & overrides"):
        st.write("Detected columns:", cols)
        key_override = st.selectbox("Keyword column", options=list(df_in.columns), index=list(df_in.columns).index(cols["keyword"]) if cols["keyword"] in df_in.columns else 0)
        url_options = ["<None>"] + list(df_in.columns)
        url_default_idx = url_options.index(cols["url"]) if cols["url"] in url_options else 0
        url_override = st.selectbox("URL column (optional)", options=url_options, index=url_default_idx)
        cols["keyword"] = key_override
        cols["url"] = None if url_override == "<None>" else url_override

        # Pick metric column from numeric columns
        numeric_candidates = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        metric_sel = st.selectbox("Metric column for parent/URL 'highest metric' logic", options=["<Auto>"] + numeric_candidates, index=0)
        metric_col_final = None if metric_sel == "<Auto>" else metric_sel

    cfg = ClusterConfig(
        model=model,
        min_similarity=min_sim,
        batch_thresholds=batch_thresholds,
        parent_strategy=parent_strategy,
        parent_metric_col=parent_metric_col_manual.strip() or metric_col_final,
        intent=IntentConfig(info_pattern=info_pat, nav_pattern=nav_pat, trans_pattern=trans_pat),
        cluster_url_strategy=cluster_url_strategy,
        cluster_separately_by_url=cluster_separately_by_url
    )

    run = st.button("🚀 Run clustering", type="primary", use_container_width=True)
    if run:
        with st.spinner("Clustering keywords..."):
            try:
                if cfg.cluster_separately_by_url:
                    results = run_pipeline_grouped_by_url(df_in, cfg, cols)
                else:
                    results = run_pipeline(df_in, cfg, cols)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        st.success("Done! Browse results per threshold below.")
        tabs = st.tabs([f"≥ {thr:.2f}" for thr in results.keys()])
        for tab, (thr, df_out) in zip(tabs, results.items()):
            with tab:
                st.markdown(f"**Rows:** {len(df_out):,}")
                st.dataframe(df_out.head(500), use_container_width=True)

                # Intent distribution
                st.markdown("#### Intent distribution")
                intent_counts = df_out["intent"].value_counts(dropna=False).reset_index()
                intent_counts.columns = ["intent", "count"]
                st.dataframe(intent_counts, use_container_width=True)

                # Top clusters by size
                st.markdown("#### Top clusters (by size)")
                cluster_sizes = df_out.groupby("parent")["keyword"].count().sort_values(ascending=False).head(20)
                st.dataframe(cluster_sizes.to_frame("count"), use_container_width=True)

                # Download
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(f"⬇️ Download CSV (≥ {thr:.2f})", data=csv, file_name=f"keywords_clustered_{thr:.2f}.csv", mime="text/csv")
st.markdown("---")
st.caption("Built with Streamlit + PolyFuzz.")
