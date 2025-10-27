
import io
import math
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import chardet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

st.set_page_config(page_title="Simple Keyword Grouper", page_icon="🧩", layout="wide")

FRIENDLY_LABELS = {
    "_label": "Cluster ID",
    "parent": "Cluster Label",
    "cluster_url": "Cluster URL",
    "url_bucket": "URL Bucket",
    "keyword": "Keyword (normalized)",
    "cluster_total_volume": "Cluster Total Volume",
    "cluster_avg_difficulty": "Cluster Avg Difficulty",
}

COLUMN_NORMALIZATION_MAP = {
    "Current position": "Position",
    "Current URL": "URL",
    "Current URL inside": "Page URL inside",
    "Current traffic": "Traffic",
    "KD": "Difficulty",
    "Keyword Difficulty": "Difficulty",
    "Search Volume": "Volume",
    "page": "URL",
    "query": "Keyword",
    "Top queries": "Keyword",
    "Impressions": "Impressions",
    "Clicks": "Clicks",
    "Search term": "Keyword",
    "Impr.": "Impressions",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df.rename(columns=COLUMN_NORMALIZATION_MAP, inplace=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def detect_encoding_from_bytes(b: bytes) -> str:
    res = chardet.detect(b)
    return res.get("encoding") or "utf-8"

def robust_read(uploaded, delimiter_opt="Auto (detect)", quotechar='"', header_row="Infer", skip_rows=0):
    name = uploaded.name.lower()
    data = uploaded.read()
    enc = detect_encoding_from_bytes(data)
    header = "infer" if header_row == "Infer" else 0
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    cand = [None, ",", "\\t", ";", "|"]
    last_err = None
    for sep in cand:
        for engine in ["python", "c"]:
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep, engine=engine,
                                 quotechar=(quotechar or '"'), skipinitialspace=True,
                                 header=header, skiprows=skip_rows, on_bad_lines="skip")
                return df
            except Exception as e:
                last_err = e
                continue
    raise ValueError(f"Failed to read file. Last error: {last_err}")

def normalize_keywords(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\\s+", " ", regex=True).str.lower()

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed — falling back to TF-IDF.")
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def vectorize(texts: List[str], use_embeddings: bool) -> np.ndarray:
    if use_embeddings:
        try:
            return embed_texts(texts)
        except Exception:
            pass
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    return vec.fit_transform(texts).toarray()

def cluster_texts_safe(X: np.ndarray, distance_threshold: float) -> np.ndarray:
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    if not np.all(np.isfinite(X)):
        return np.arange(n, dtype=int)
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold
        )
        return model.fit_predict(X)
    except Exception:
        return np.arange(n, dtype=int)

def label_centroids_tfidf(keywords: List[str], labels: np.ndarray) -> Dict[int, str]:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(keywords).toarray()
    reps = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) == 0:
            continue
        centroid = X[idx].mean(axis=0, keepdims=True)
        sims = (X[idx] @ centroid.T).ravel()
        reps[int(lbl)] = keywords[idx[int(np.argmax(sims))]]
    return reps

def postprocess_min_max(out: pd.DataFrame, min_size: int, max_size: int, use_embeddings: bool, dist_thr: float) -> pd.DataFrame:
    """
    Enforce per-bucket min/max cluster sizes within a block.
    - min_size: clusters smaller than this become singletons (each row its own label).
    - max_size (>0): clusters larger than this are split by KMeans into ceil(size/max_size) parts.
    """
    if out.empty:
        return out
    # Work with a copy
    df = out.copy()
    base_labels = df["_label"].values
    keywords = df["keyword"].tolist()
    next_label = int(base_labels.max()) + 1 if len(df) else 0

    # Build a stable TF-IDF space once for labeling/splitting
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_tfidf = vec.fit_transform(keywords).toarray()

    new_labels = base_labels.copy()

    for lbl in np.unique(base_labels):
        idx = np.where(base_labels == lbl)[0]
        size = len(idx)
        if size == 0:
            continue

        # Too small → singletons
        if size < max(1, int(min_size)):
            for i in idx:
                new_labels[i] = next_label
                next_label += 1
            continue

        # Too big → split if max_size is set
        if max_size and max_size > 0 and size > max_size:
            k = math.ceil(size / max_size)
            k = max(2, min(k, size))  # ensure 2 <= k <= size
            # Split with KMeans in TF-IDF space for stability
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            sub_X = X_tfidf[idx]
            sub_labels = km.fit_predict(sub_X)
            # Remap to global labels
            for sub_k in np.unique(sub_labels):
                sub_idx = idx[np.where(sub_labels == sub_k)[0]]
                for i in sub_idx:
                    new_labels[i] = next_label
                next_label += 1

    df["_label"] = new_labels

    # Recompute representative names per final label
    reps = {}
    for lbl in np.unique(new_labels):
        sub_idx = np.where(new_labels == lbl)[0]
        centroid = X_tfidf[sub_idx].mean(axis=0, keepdims=True)
        sims = (X_tfidf[sub_idx] @ centroid.T).ravel()
        reps[int(lbl)] = keywords[sub_idx[int(np.argmax(sims))]]
    df["parent"] = df["_label"].map(reps)

    return df

st.title("🧩 Simple Keyword Grouper")
st.caption("Group by URL or cluster keywords. Optional cluster totals & averages with dropdown-selected metrics. Per-URL min/max cluster size.")

with st.sidebar:
    st.header("1) Data")
    upl = st.file_uploader("Upload CSV/TSV/XLSX", type=["csv", "tsv", "xlsx", "xls"])
    use_example = st.toggle("Use example data", value=False)

    st.header("2) Grouping mode")
    mode = st.selectbox(
        "Choose how to group:",
        [
            "Group by existing URL column (cluster within each URL)",
            "Use a single provided URL (cluster all keywords)",
            "Keywords only (ignore URLs; cluster all)",
        ],
        index=0
    )

    st.header("3) Clustering")
    dist_thr = st.slider("Clustering strictness (cosine distance threshold)", 0.05, 1.0, 0.25, 0.01)
    use_embeddings = st.toggle("Use embeddings (if installed) instead of TF-IDF", value=False)

    st.header("4) Per-URL cluster size constraints")
    min_cluster_size = st.number_input("Minimum cluster size", min_value=1, max_value=1000, value=5, help="Clusters smaller than this become singletons (per URL bucket).")
    max_cluster_size = st.number_input("Maximum cluster size (0 = no limit)", min_value=0, max_value=1000, value=20, help="Clusters larger than this are automatically split (per URL bucket).")

    provided_url = ""
    if mode == "Use a single provided URL (cluster all keywords)":
        provided_url = st.text_input("Target URL to assign to clusters", value="")

    st.header("5) Metrics (optional)")
    # Will populate after data load

    st.markdown("---")
    run_btn = st.button("🚀 Run", type="primary", use_container_width=True)

# Load data
if use_example:
    df = pd.DataFrame({
        "Keyword": ["best running shoes", "buy running shoes", "what are trail shoes",
                    "nike pegasus review", "asics gel-kayano price", "trail runners vs hiking shoes",
                    "adidas ultraboost sale", "running shoe size guide", "brooks ghost 15", "shoe store near me"],
        "URL": ["/shoes/best", "/shoes/buy", "/shoes/trail", "/shoes/pegasus", "/shoes/gel-kayano",
                "/shoes/trail", "/shoes/ultraboost", "/guides/sizing", "/shoes/brooks-ghost", "/stores/near-me"],
        "Volume": [12000, 9000, 4000, 3000, 2500, 1500, 8000, 2000, 1800, 10000],
        "Difficulty": [55, 50, 35, 65, 60, 40, 45, 30, 50, 25]
    })
    st.info("Using example data.")
elif upl is not None:
    try:
        df = robust_read(upl)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    df = None

if df is not None:
    df = normalize_columns(df)

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Column mapping
    with st.expander("Column mapping (optional)", expanded=False):
        cols = list(df.columns)
        key_col = st.selectbox("Keyword column", options=cols, index=(cols.index("Keyword") if "Keyword" in cols else 0))
        url_col = st.selectbox("URL column (optional)", options=["<None>"] + cols,
                               index=(["<None>"] + cols).index("URL") if "URL" in cols else 0)

    # Metrics dropdowns
    cols_all = ["<None>"] + list(df.columns)
    vol_default = "<None>"
    for cand in ["Volume", "Search Volume", "Impressions", "Traffic", "Clicks"]:
        if cand in df.columns:
            vol_default = cand
            break
    diff_default = "<None>"
    for cand in ["Difficulty", "KD", "Keyword Difficulty"]:
        if cand in df.columns:
            diff_default = cand
            break
    vol_col = st.sidebar.selectbox("Volume / Proxy (sum per cluster)", options=cols_all, index=cols_all.index(vol_default))
    diff_col = st.sidebar.selectbox("Difficulty (avg per cluster)", options=cols_all, index=cols_all.index(diff_default))

    # Prep data
    df_work = df.copy()
    if key_col not in df_work.columns:
        st.error("Selected Keyword column not found.")
        st.stop()
    df_work.rename(columns={key_col: "Keyword"}, inplace=True)
    df_work["keyword"] = df_work["Keyword"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

    url_col_final = None if url_col == "<None>" else url_col
    if mode == "Group by existing URL column (cluster within each URL)":
        if not url_col_final or url_col_final not in df_work.columns:
            st.warning("No URL column selected. Switch mode or choose a URL column in 'Column mapping'.")
    elif mode == "Use a single provided URL (cluster all keywords)":
        url_col_final = "URL"
        df_work[url_col_final] = provided_url.strip() if provided_url else "<no-url>"
    else:
        url_col_final = None

    # Run
    if run_btn:
        if mode == "Group by existing URL column (cluster within each URL)" and (not url_col_final or url_col_final not in df_work.columns):
            st.error("URL column required for this mode.")
            st.stop()

        def cluster_block(block: pd.DataFrame) -> pd.DataFrame:
            n = len(block)
            if n == 0:
                return block.assign(_label=np.array([], dtype=int), parent=np.array([], dtype=str))
            texts = block["keyword"].tolist()
            X = vectorize(texts, use_embeddings=use_embeddings)
            labels = cluster_texts_safe(X, distance_threshold=float(dist_thr))
            out = block.copy()
            out["_label"] = labels
            # initial centroid labels (TF-IDF for stability)
            reps = label_centroids_tfidf(texts, labels)
            out["parent"] = out["_label"].map(reps)
            # apply per-bucket min/max
            out = postprocess_min_max(out, int(min_cluster_size), int(max_cluster_size), use_embeddings, float(dist_thr))
            return out

        if mode == "Group by existing URL column (cluster within each URL)" and url_col_final:
            outputs = []
            for val, sub in df_work.groupby(url_col_final, dropna=False):
                res = cluster_block(sub)
                res["url_bucket"] = val
                res["cluster_url"] = val
                outputs.append(res)
            df_out = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
        else:
            df_out = cluster_block(df_work)
            if url_col_final and url_col_final in df_work.columns:
                canon = {}
                for p, sub in df_out.groupby("parent", sort=False):
                    canon[p] = df_out[url_col_final].mode().iloc[0] if not df_out[url_col_final].isna().all() else None
                df_out["cluster_url"] = df_out["parent"].map(canon)
                df_out["url_bucket"] = df_out[url_col_final]
            else:
                df_out["cluster_url"] = None
                df_out["url_bucket"] = None

        # Cluster-level metrics
        def safe_num(s):
            return pd.to_numeric(s, errors="coerce")

        if not df_out.empty and vol_col and vol_col != "<None>" and vol_col in df_out.columns:
            vol_series = safe_num(df_out[vol_col]).fillna(0.0)
            cluster_total_volume = vol_series.groupby(df_out["parent"]).sum()
            df_out["cluster_total_volume"] = df_out["parent"].map(cluster_total_volume)
        else:
            df_out["cluster_total_volume"] = np.nan

        if not df_out.empty and diff_col and diff_col != "<None>" and diff_col in df_out.columns:
            diff_series = safe_num(df_out[diff_col])
            cluster_avg_difficulty = diff_series.groupby(df_out["parent"]).mean()
            df_out["cluster_avg_difficulty"] = df_out["parent"].map(cluster_avg_difficulty)
        else:
            df_out["cluster_avg_difficulty"] = np.nan

        # Output
        st.subheader("Results")
        preview = df_out.copy()
        if "Keyword" in preview.columns and "keyword" in preview.columns:
            preview = preview.drop(columns=["Keyword"])
        preview = preview.rename(columns=FRIENDLY_LABELS)
        ordered = ["Keyword (normalized)", "Cluster Label", "Cluster Total Volume", "Cluster Avg Difficulty",
                   "Cluster URL", "URL Bucket", "Cluster ID"]
        cols_existing = [c for c in ordered if c in preview.columns] + [c for c in preview.columns if c not in ordered]
        preview = preview[cols_existing]
        preview = preview.loc[:, ~preview.columns.duplicated()]
        st.dataframe(preview.head(1000), use_container_width=True)

        # Export CSV
        export_df = preview.copy()
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name="keywords_grouped_with_minmax_metrics.csv", mime="text/csv")

st.markdown("---")
st.caption("Per-URL min/max cluster size applied after clustering. Small clusters become singletons; large clusters split into subclusters.")
