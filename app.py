
import io
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import chardet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

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
    "Impressions": "Volume",
    "Clicks": "Traffic",
    "Search term": "Keyword",
    "Impr.": "Volume",
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

def cluster_texts(X: np.ndarray, distance_threshold: float) -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold
    )
    labels = model.fit_predict(X)
    return labels

def label_by_centroid(keywords: List[str], X: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
    out = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) == 0:
            continue
        centroid = X[idx].mean(axis=0, keepdims=True)
        sims = (X[idx] @ centroid.T).ravel()
        out[int(lbl)] = keywords[idx[np.argmax(sims)]]
    return out

st.title("🧩 Simple Keyword Grouper")
st.caption("Straightforward grouping & clustering by URL or keywords. Minimal options for consistent results.")

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

    st.header("3) Options")
    dist_thr = st.slider("Clustering strictness (cosine distance threshold)", 0.05, 1.0, 0.25, 0.01)
    use_embeddings = st.toggle("Use embeddings (if installed) instead of TF-IDF", value=False)

    provided_url = ""
    if mode == "Use a single provided URL (cluster all keywords)":
        provided_url = st.text_input("Target URL to assign to clusters", value="")

    st.markdown("---")
    run_btn = st.button("🚀 Run", type="primary", use_container_width=True)

if use_example:
    df = pd.DataFrame({
        "Keyword": ["best running shoes", "buy running shoes", "what are trail shoes",
                    "nike pegasus review", "asics gel-kayano price", "trail runners vs hiking shoes",
                    "adidas ultraboost sale", "running shoe size guide", "brooks ghost 15", "shoe store near me"],
        "URL": ["/shoes/best", "/shoes/buy", "/shoes/trail", "/shoes/pegasus", "/shoes/gel-kayano",
                "/shoes/trail", "/shoes/ultraboost", "/guides/sizing", "/shoes/brooks-ghost", "/stores/near-me"]
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

    with st.expander("Column mapping (optional)", expanded=False):
        cols = list(df.columns)
        key_col = st.selectbox("Keyword column", options=cols, index=(cols.index("Keyword") if "Keyword" in cols else 0))
        url_col = st.selectbox("URL column (optional)", options=["<None>"] + cols,
                               index=(["<None>"] + cols).index("URL") if "URL" in cols else 0)

    df_work = df.copy()
    if key_col not in df_work.columns:
        st.error("Selected Keyword column not found.")
        st.stop()
    df_work.rename(columns={key_col: "Keyword"}, inplace=True)
    df_work["keyword"] = df_work["Keyword"].astype(str)
    df_work["keyword"] = df_work["keyword"].str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

    url_col_final = None if url_col == "<None>" else url_col
    if mode == "Group by existing URL column (cluster within each URL)":
        if not url_col_final or url_col_final not in df_work.columns:
            st.warning("No URL column selected. Switch mode or choose a URL column in 'Column mapping'.")
    elif mode == "Use a single provided URL (cluster all keywords)":
        url_col_final = "URL"
        df_work[url_col_final] = provided_url.strip() if provided_url else "<no-url>"
    else:
        url_col_final = None

    if run_btn:
        if mode == "Group by existing URL column (cluster within each URL)" and (not url_col_final or url_col_final not in df_work.columns):
            st.error("URL column required for this mode.")
            st.stop()

        def cluster_block(block: pd.DataFrame) -> pd.DataFrame:
            texts = block["keyword"].tolist()
            X = vectorize(texts, use_embeddings=use_embeddings)
            labels = cluster_texts(X, distance_threshold=float(dist_thr))
            lbl2name = label_by_centroid(texts, X, labels)
            out = block.copy()
            out["_label"] = labels
            out["parent"] = out["_label"].map(lbl2name)
            return out

        if mode == "Group by existing URL column (cluster within each URL)" and url_col_final:
            outputs = []
            for val, sub in df_work.groupby(url_col_final, dropna=False):
                res = cluster_block(sub)
                res["url_bucket"] = val
                res["cluster_url"] = val
                outputs.append(res)
            df_out = pd.concat(outputs, ignore_index=True)
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

        st.subheader("Results")
        preview = df_out.copy()
        if "Keyword" in preview.columns and "keyword" in preview.columns:
            preview = preview.drop(columns=["Keyword"])
        preview = preview.rename(columns=FRIENDLY_LABELS)
        preview = preview.loc[:, ~preview.columns.duplicated()]
        st.dataframe(preview.head(1000), use_container_width=True)

        export_cols = ["keyword", "parent", "_label", "cluster_url", "url_bucket"]
        export_cols = [c for c in export_cols if c in df_out.columns]
        export_df = df_out[export_cols + (["Keyword"] if "Keyword" in df_out.columns else [])].copy()
        export_df = export_df.rename(columns=FRIENDLY_LABELS)
        export_df = export_df.loc[:, ~export_df.columns.duplicated()]
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name="keywords_grouped.csv", mime="text/csv")

st.markdown("---")
st.caption("For consistent results, leave embeddings OFF (TF-IDF) and use the default threshold (0.25).")
