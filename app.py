
import io
import re
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import chardet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Optional deps
try:
    import hdbscan
except Exception:
    hdbscan = None

try:
    import networkx as nx
    from community import community_louvain  # python-louvain
except Exception:
    nx = None
    community_louvain = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from polyfuzz import PolyFuzz
except Exception:
    PolyFuzz = None

st.set_page_config(page_title="Keyword Clustering (SEO)", page_icon="🔎", layout="wide")

# --- persistent override state ---
if "overrides" not in st.session_state:
    st.session_state.overrides = {
        "rename": {},        # {old_label -> new_label}
        "move": {},          # {keyword -> dest_label}
        "must_link": [],     # [(kw1, kw2), ...]
        "cannot_link": [],   # [(kw1, kw2), ...]
    }

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
    # Representation
    representation: str = "SBERT (MiniLM)"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Algorithm
    algorithm: str = "HDBSCAN"  # KMeans, Agglomerative, HDBSCAN, Graph (Louvain), PolyFuzz
    # Common options
    batch_thresholds: Tuple[float, ...] = (0.6, 0.7, 0.8)  # for PolyFuzz
    k: int = 20  # KMeans clusters
    distance_threshold: float = 0.25  # Agglomerative (cosine distance)
    hdb_min_cluster_size: int = 5
    hdb_min_samples: Optional[int] = None
    graph_sim_cutoff: float = 0.70
    # Labeling
    parent_strategy: str = "Highest metric"  # Highest metric / Centroid keyword / TF-IDF phrase / Longest / Shortest / Representative
    parent_metric_col: Optional[str] = None
    tfidf_top_k: int = 2
    # Columns
    keyword_col_candidates: Tuple[str, ...] = (
        "Keyword", "Keywords", "Query", "Queries", "Search term", "Search terms", "Term"
    )
    url_col_candidates: Tuple[str, ...] = (
        "URL", "Page", "Page URL", "Landing Page", "Final URL", "Destination URL", "Canonical", "Slug"
    )
    impressions_col_candidates: Tuple[str, ...] = (
        "Impressions", "Clicks", "Sessions", "Pageviews", "Volume", "Search Volume", "SV", "Traffic"
    )
    intent: IntentConfig = field(default_factory=IntentConfig)
    # URL controls
    cluster_separately_by_url: bool = False
    max_keywords_per_url: Optional[int] = None  # per-bucket cap

# -----------------------------
# Robust file reading
# -----------------------------
def detect_encoding_from_bytes(b: bytes) -> str:
    res = chardet.detect(b)
    return res.get("encoding") or "utf-8"

def _try_read_csv(bytes_buf: bytes, *, encoding: str, sep, engine, on_bad_lines, quotechar, header, skiprows, thousands, decimal):
    return pd.read_csv(
        io.BytesIO(bytes_buf),
        encoding=encoding,
        sep=sep,
        engine=engine,
        on_bad_lines=on_bad_lines,
        quotechar=quotechar if quotechar else '"',
        skipinitialspace=True,
        header=header,
        skiprows=skiprows if skiprows else None,
        thousands=thousands if thousands else None,
        decimal=decimal if decimal else "."
    )

def read_table_from_upload(f, *, delimiter_opt="Auto (detect)", quotechar='"', bad_line_behavior="error",
                           use_python_engine=True, header_row="Infer", skip_rows=0, thousands="", decimal=".") -> pd.DataFrame:
    """
    Flexible reader:
    - Auto detects encoding (chardet)
    - Auto/sniff delimiter; fallbacks through comma, tab, semicolon, pipe
    - Can skip/warn bad lines
    - Lets you pick header inference and row skipping
    """
    name = f.name.lower()
    content = f.read()
    encoding = detect_encoding_from_bytes(content)

    header = "infer" if header_row == "Infer" else (int(header_row) if isinstance(header_row, int) else 0)
    on_bad = None if bad_line_behavior == "error" else bad_line_behavior

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))

    # CSV/TSV and unknown
    # Map delimiter
    sep_map = {
        "Auto (detect)": None,
        ",": ",",
        "\\t": "\t",
        ";": ";",
        "|": "|"
    }
    sep = sep_map.get(delimiter_opt, None)
    # Try candidates in order
    candidates = []
    if sep is None:
        candidates = [None, ",", "\t", ";", "|"]
    else:
        candidates = [sep]

    # engine choice
    engines = ["python"] if (use_python_engine or sep is None) else ["c", "python"]

    last_err = None
    for eng in engines:
        for s in candidates:
            try:
                return _try_read_csv(content, encoding=encoding, sep=s, engine=eng, on_bad_lines=on_bad,
                                     quotechar=quotechar, header=header, skiprows=skip_rows, thousands=thousands, decimal=decimal)
            except Exception as e:
                last_err = e
                continue

    # Final fallback: force python engine + autodetect + warn on bad lines
    try:
        return _try_read_csv(content, encoding=encoding, sep=None, engine="python", on_bad_lines="warn",
                             quotechar=quotechar, header=header, skiprows=skip_rows, thousands=thousands, decimal=decimal)
    except Exception as e:
        raise ValueError(f"Failed to read file robustly. Last error: {last_err or e}")

# -----------------------------
# Column detection
# -----------------------------
def _find_first(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    # exact lowercase match
    for cand in candidates:
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    # contains match
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    # heuristic: keyword-like: shortest text-ish column
    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    if text_cols:
        return text_cols[0]
    return None

def detect_columns(df: pd.DataFrame, cfg: ClusterConfig) -> Dict[str, Optional[str]]:
    key_col = _find_first(df, cfg.keyword_col_candidates)
    url_col = _find_first(df, cfg.url_col_candidates)
    # Prefer a numeric metric if present; otherwise first candidate
    metric_col = None
    # first: candidate list overlap that is numeric
    for c in df.columns:
        if any(tok.lower() in c.lower() for tok in cfg.impressions_col_candidates):
            if pd.api.types.is_numeric_dtype(df[c]):
                metric_col = c
                break
    # fallback: any numeric column
    if metric_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                metric_col = c
                break
    return {"keyword": key_col, "url": url_col, "metric": metric_col}

def normalize_keywords(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower())

# -----------------------------
# Representations
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_sbert(model_name: str):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required. Add to requirements and reinstall.")
    return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = load_sbert(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def vectorize_tfidf(texts: List[str]) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    return vec.fit_transform(texts).toarray()

# -----------------------------
# Algorithms
# -----------------------------
def algo_kmeans(X: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    return labels, {"silhouette": float(score)}

def algo_agglomerative(X: np.ndarray, distance_threshold: float):
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=distance_threshold
    )
    labels = model.fit_predict(X)
    return labels, {}

def algo_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: Optional[int]):
    if hdbscan is None:
        raise ImportError("hdbscan is required. Add to requirements and reinstall.")
    cl = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0
    )
    labels = cl.fit_predict(X)
    meta = {"noise": int(np.sum(labels == -1))}
    return labels, meta

def algo_graph_louvain(X: np.ndarray, sim_cutoff: float):
    if nx is None or community_louvain is None:
        raise ImportError("networkx and python-louvain are required. Add to requirements.")
    sims = cosine_similarity(X)
    np.fill_diagonal(sims, 0.0)
    rows, cols = np.where(sims >= sim_cutoff)
    G = nx.Graph()
    G.add_nodes_from(range(X.shape[0]))
    G.add_edges_from((int(i), int(j), {"weight": float(sims[i,j])}) for i,j in zip(rows, cols) if i<j)
    if G.number_of_edges() == 0:
        return np.arange(X.shape[0]), {"note": "no edges at cutoff"}
    parts = community_louvain.best_partition(G, weight="weight", random_state=42)
    labels = np.array([parts[i] for i in range(X.shape[0])], dtype=int)
    return labels, {}

@st.cache_data(show_spinner=False)
def polyfuzz_match(keywords: List[str], model_name: str = "TF-IDF") -> pd.DataFrame:
    if PolyFuzz is None:
        raise ImportError("polyfuzz is required. Add to requirements and reinstall.")
    model = PolyFuzz(model_name)
    model.match(keywords, keywords)
    m = model.get_matches().rename(columns={"From":"keyword","To":"cluster_repr","Similarity":"similarity"})
    return m

# -----------------------------
# Labeling
# -----------------------------
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

def label_by_tfidf_phrase(keywords: List[str], labels: np.ndarray, top_k: int=2) -> Dict[int, str]:
    out = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) == 0:
            continue
        texts = [keywords[i] for i in idx]
        vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english").fit(texts)
        tf = vec.transform(texts).sum(axis=0).A1
        terms = np.array(vec.get_feature_names_out())
        top = terms[tf.argsort()[::-1][:top_k]]
        out[int(lbl)] = " ".join(top)
    return out

def label_by_metric(df: pd.DataFrame, labels: np.ndarray, metric_col: Optional[str]) -> Dict[int, str]:
    """
    Pick the parent keyword per cluster by the highest value in `metric_col`.
    Falls back to the first keyword if the metric column is missing/empty.
    Uses a position-safe argmax to avoid index misalignment.
    """
    out: Dict[int, str] = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) == 0:
            continue  # empty cluster edge-case

        sub = df.iloc[idx]  # subset frame
        if metric_col and metric_col in sub.columns:
            ser = pd.to_numeric(sub[metric_col], errors="coerce").fillna(0)
            if ser.size == 0:
                out[int(lbl)] = sub["keyword"].iloc[0]
            else:
                pos = int(np.nanargmax(ser.to_numpy()))  # positional argmax within subset
                out[int(lbl)] = sub["keyword"].iloc[pos]
        else:
            out[int(lbl)] = sub["keyword"].iloc[0]
    return out

# -----------------------------
# Pipeline helpers
# -----------------------------
def stratified_cap_by_url(df: pd.DataFrame, url_col: Optional[str], cap: Optional[int]) -> pd.DataFrame:
    # Robust guards
    if df is None or df.empty:
        return df
    if not url_col or url_col not in df.columns or cap is None:
        return df
    try:
        cap = int(cap)
    except Exception:
        return df
    if cap <= 0:
        return df

    parts = []
    try:
        for url, sub in df.groupby(url_col, dropna=False):
            if len(sub) > cap:
                parts.append(sub.sample(n=cap, random_state=42))
            else:
                parts.append(sub)
        return pd.concat(parts, ignore_index=True)
    except Exception as e:
        st.warning(f"Per-URL cap skipped due to: {e}")
        return df

def build_representation_texts(keywords: List[str], cfg: ClusterConfig) -> np.ndarray:
    if cfg.representation == "TF-IDF":
        return vectorize_tfidf(keywords)
    elif cfg.representation.startswith("SBERT"):
        model = cfg.sbert_model if "MiniLM" in cfg.representation else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        return embed_texts(keywords, model)
    else:
        return vectorize_tfidf(keywords)

def run_algo(X: np.ndarray, cfg: ClusterConfig):
    if cfg.algorithm == "KMeans":
        return algo_kmeans(X, cfg.k)
    elif cfg.algorithm == "Agglomerative":
        return algo_agglomerative(X, cfg.distance_threshold)
    elif cfg.algorithm == "HDBSCAN":
        return algo_hdbscan(X, cfg.hdb_min_cluster_size, cfg.hdb_min_samples)
    elif cfg.algorithm == "Graph (Louvain)":
        return algo_graph_louvain(X, cfg.graph_sim_cutoff)
    else:
        raise NotImplementedError

def compute_umap_2d(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
        return reducer.fit_transform(X)
    except Exception as e:
        raise RuntimeError(f"UMAP unavailable: {e}")

def suggest_k(n_rows: int, desired_size: int) -> int:
    return max(2, int(round(n_rows / max(2, desired_size))))

# -----------------------------
# UI
# -----------------------------
st.title("🔎 Keyword Clustering for SEO")
st.caption("Robust CSV import • Flexible column mapping • Embeddings & classic clustering • URL-aware controls • Cluster Manager")

with st.sidebar:
    st.header("1) Data")
    uploaded = st.file_uploader("Upload CSV/TSV/XLSX", type=["csv", "tsv", "xlsx", "xls"])
    use_example = st.toggle("Use example data", value=False)

    st.header("2) Parsing options")
    delimiter_opt = st.selectbox("Delimiter", ["Auto (detect)", ",", "\\t", ";", "|"], index=0)
    quotechar = st.text_input("Quote character", '"')
    bad_line_behavior = st.selectbox("If malformed rows:", ["error", "skip", "warn"], index=0)
    use_python_engine = st.toggle("Use Python engine (more forgiving)", value=True)
    header_row = st.selectbox("Header row", ["Infer", "0 (first row is header)"], index=0)
    skip_rows = st.number_input("Skip top rows", min_value=0, max_value=10000, value=0)
    thousands = st.text_input("Thousands separator (optional)", "")
    decimal = st.text_input("Decimal separator (default .)", ".")

    st.header("3) Representation")
    representation = st.selectbox("Text representation", ["SBERT (MiniLM)", "SBERT (Multilingual)", "TF-IDF"], index=0)

    st.header("4) Algorithm")
    algorithm = st.selectbox("Clustering algorithm", ["HDBSCAN", "KMeans", "Agglomerative", "Graph (Louvain)", "PolyFuzz (string match batches)"], index=0)

    st.header("5) URL-level controls")
    cap_per_url = st.number_input("Max keywords per URL (cap per bucket)", min_value=1, max_value=100000, value=500)
    cluster_by_url = st.toggle("Cluster separately per URL (bucket first)", value=False)

    st.header("6) Labeling")
    label_strategy = st.selectbox("Primary label shown", ["Highest metric", "Centroid keyword", "TF-IDF phrase", "Longest", "Shortest", "Representative"], index=0)
    tfidf_top_k = st.slider("TF-IDF phrase words (n-grams)", 1, 4, 2)

    st.header("7) Algo parameters")
    k_val = st.number_input("KMeans: k", min_value=2, max_value=2000, value=20)
    dist_thr = st.slider("Agglomerative: distance threshold (cosine)", 0.0, 1.5, 0.25, 0.01)
    hdb_min_size = st.number_input("HDBSCAN: min cluster size", min_value=2, max_value=500, value=5)
    hdb_min_samples = st.number_input("HDBSCAN: min samples (0 = auto)", min_value=0, max_value=500, value=0)
    graph_cut = st.slider("Graph/Louvain: similarity cutoff", 0.0, 1.0, 0.70, 0.01)

    st.header("8) Cluster size preview (for KMeans)")
    desired_avg = st.number_input("Desired avg keywords per cluster", min_value=2, max_value=2000, value=50)

    st.markdown("---")
    run_btn = st.button("🚀 Run clustering", type="primary", use_container_width=True)

# Load data
if use_example:
    df_in = pd.DataFrame({
        "Keyword": ["best running shoes", "buy running shoes", "what are trail shoes",
                    "nike pegasus review", "asics gel-kayano price", "trail runners vs hiking shoes",
                    "adidas ultraboost sale", "running shoe size guide", "brooks ghost 15", "shoe store near me"],
        "Impressions": [1000, 800, 600, 500, 450, 400, 700, 300, 350, 900],
        "URL": [" /shoes/best " , "/shoes/buy", "/shoes/trail", "/shoes/pegasus", "/shoes/gel-kayano",
                "/shoes/trail", "/shoes/ultraboost", "/guides/sizing", "/shoes/brooks-ghost", "/stores/near-me"]
    })
    st.info("Using example data.")
elif uploaded is not None:
    try:
        df_in = read_table_from_upload(
            uploaded,
            delimiter_opt=delimiter_opt,
            quotechar=quotechar,
            bad_line_behavior=bad_line_behavior,
            use_python_engine=use_python_engine,
            header_row=header_row,
            skip_rows=int(skip_rows),
            thousands=thousands,
            decimal=decimal
        )
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    df_in = None

if df_in is not None:
    st.subheader("Preview input")
    st.dataframe(df_in.head(20), use_container_width=True)
    st.caption(f"Columns detected: {list(df_in.columns)}")

    # Detect columns and override controls
    cols_detect = detect_columns(df_in, ClusterConfig())
    with st.expander("Column mapping", expanded=False):
        st.write("Detected:", cols_detect)
        # Keyword
        key_col = st.selectbox("Keyword column", options=list(df_in.columns), index=(list(df_in.columns).index(cols_detect["keyword"]) if cols_detect["keyword"] in df_in.columns else 0))
        # URL
        url_options = ["<None>"] + list(df_in.columns)
        url_default_idx = url_options.index(cols_detect["url"]) if (cols_detect["url"] in url_options) else 0
        url_col_sel = st.selectbox("URL column (optional)", options=url_options, index=url_default_idx)
        # Metric
        numeric_cols = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        metric_candidates = ["<Auto>"] + numeric_cols + [c for c in df_in.columns if c not in numeric_cols]
        metric_col = st.selectbox("Metric column (for Highest metric labeling)", options=metric_candidates, index=0)

    # Normalize and cap per URL
    df_work = df_in.copy()
    if key_col not in df_work.columns:
        st.error("Selected keyword column not found in data.")
        st.stop()
    df_work.rename(columns={key_col: "Keyword"}, inplace=True)
    df_work["keyword"] = normalize_keywords(df_work["Keyword"])

    url_col_final = None if url_col_sel == "<None>" else url_col_sel
    if url_col_final and url_col_final not in df_work.columns:
        st.warning(f"Selected URL column '{url_col_final}' not found in data. Skipping per-URL capping and URL grouping.")
        url_col_final = None
    if url_col_final:
        df_work[url_col_final] = df_work[url_col_final].astype(str).str.strip()

    n_before = len(df_work)
    df_work = stratified_cap_by_url(df_work, url_col_final, cap_per_url)
    st.markdown(f"**Rows after per-URL cap:** {len(df_work):,} (from {n_before:,})")

    # KMeans suggestion
    if algorithm == "KMeans":
        st.caption(f"Suggested k ≈ {suggest_k(len(df_work), int(desired_avg))} (you set k={k_val})")

    # Build cfg
    cfg = ClusterConfig(
        representation=representation,
        algorithm="PolyFuzz" if algorithm.startswith("PolyFuzz") else algorithm,
        batch_thresholds=tuple(sorted(set(float(x.strip()) for x in str(','.join([str(x) for x in ([] if isinstance(getattr(ClusterConfig, 'batch_thresholds', ()), str) else [])])) if x.strip()))),
        k=int(k_val),
        distance_threshold=float(dist_thr),
        hdb_min_cluster_size=int(hdb_min_size),
        hdb_min_samples=None if int(hdb_min_samples)==0 else int(hdb_min_samples),
        graph_sim_cutoff=float(graph_cut),
        parent_strategy=label_strategy,
        parent_metric_col=None if metric_col=="<Auto>" else metric_col,
        tfidf_top_k=int(tfidf_top_k),
        cluster_separately_by_url=bool(cluster_by_url),
        max_keywords_per_url=int(cap_per_url)
    )
    # Fix batch_thresholds to what user typed in earlier sidebar (can't reuse earlier code block here)
    # We'll parse a simple text input for thresholds for PolyFuzz; since we lost it above, we set a default here.
    # Add a small input for thresholds here too for safety:
    if algorithm.startswith("PolyFuzz"):
        poly_thr_text = st.text_input("PolyFuzz thresholds (comma-separated)", "0.6,0.7,0.8", key="poly_thr_in_block")
        try:
            cfg.batch_thresholds = tuple(sorted({float(x.strip()) for x in poly_thr_text.split(",") if x.strip()}))
        except Exception:
            cfg.batch_thresholds = (0.6, 0.7, 0.8)

    if run_btn:
        with st.spinner("Clustering..."):
            if cfg.algorithm == "PolyFuzz":
                matches = polyfuzz_match(df_work["keyword"].tolist(), "TF-IDF")
                merged = matches.merge(df_work, on="keyword", how="left")
                thresholds = sorted(cfg.batch_thresholds)
                tabs = st.tabs([f"PolyFuzz ≥ {t:.2f}" for t in thresholds])
                for tab, thr in zip(tabs, thresholds):
                    with tab:
                        temp = merged[merged["similarity"] >= thr].copy()
                        temp["parent_representative"] = temp["cluster_repr"]
                        metric_c = cfg.parent_metric_col or cols_detect["metric"]
                        temp["_lbl"] = temp["cluster_repr"].astype("category").cat.codes
                        labels_metric = label_by_metric(temp.assign(keyword=temp["keyword"]), temp["_lbl"].values, metric_c)
                        temp["label_metric"] = temp["_lbl"].map(labels_metric)
                        label_col_to_use = "label_metric" if cfg.parent_strategy == "Highest metric" else "parent_representative"
                        temp["parent"] = temp[label_col_to_use]
                        # URL canon
                        if url_col_final and url_col_final in temp.columns:
                            canon = {}
                            for p, sub in temp.groupby("parent", sort=False):
                                row = sub.loc[sub["keyword"] == p]
                                if row.empty:
                                    u = sub[url_col_final].mode().iloc[0] if not sub[url_col_final].isna().all() else None
                                else:
                                    u = row[url_col_final].iloc[0]
                                canon[p] = u
                            temp["cluster_url"] = temp["parent"].map(canon)

                        n_clusters = temp["parent"].nunique()
                        sizes = temp.groupby("parent")["keyword"].count().sort_values(ascending=False)
                        st.markdown(f"**Clusters:** {n_clusters} • **Avg size:** {sizes.mean():.2f} • **Median size:** {sizes.median():.0f}")
                        st.dataframe(temp.head(500), use_container_width=True)
                        st.markdown("**Top clusters by size**")
                        st.dataframe(sizes.head(20).to_frame("count"))

                        csv = temp.to_csv(index=False).encode("utf-8")
                        st.download_button(f"⬇️ Download CSV (≥ {thr:.2f})", data=csv, file_name=f"keywords_clustered_polyfuzz_{thr:.2f}.csv", mime="text/csv")
            else:
                # Embedding/Vector path
                def cluster_block(block: pd.DataFrame):
                    keywords = block["keyword"].tolist()
                    X = build_representation_texts(keywords, cfg)
                    labels, meta = run_algo(X, cfg)
                    out = block.copy()
                    out["_label"] = labels
                    metric_c = cfg.parent_metric_col or cols_detect["metric"]
                    label_metric = label_by_metric(out, labels, metric_c)
                    label_centroid = label_by_centroid(keywords, X, labels)
                    label_tfidf = label_by_tfidf_phrase(keywords, labels, top_k=cfg.tfidf_top_k)

                    # Longest/Shortest/Representative
                    label_long = {}
                    label_short = {}
                    label_repr = {}
                    for lbl in np.unique(labels):
                        idx = np.where(labels==lbl)[0]
                        kws = np.array(keywords)[idx]
                        if len(kws)==0:
                            continue
                        lengths = np.array([len(k) for k in kws])
                        label_long[int(lbl)] = kws[int(np.argmax(lengths))]
                        label_short[int(lbl)] = kws[int(np.argmin(lengths))]
                        label_repr[int(lbl)] = label_centroid[int(lbl)]

                    def pick(lbl: int) -> str:
                        if cfg.parent_strategy == "Highest metric":
                            return label_metric.get(int(lbl))
                        elif cfg.parent_strategy == "Centroid keyword":
                            return label_centroid.get(int(lbl))
                        elif cfg.parent_strategy == "TF-IDF phrase":
                            return label_tfidf.get(int(lbl))
                        elif cfg.parent_strategy == "Longest":
                            return label_long.get(int(lbl))
                        elif cfg.parent_strategy == "Shortest":
                            return label_short.get(int(lbl))
                        else:
                            return label_repr.get(int(lbl))

                    out["label_metric"] = out["_label"].map(label_metric)
                    out["label_centroid"] = out["_label"].map(label_centroid)
                    out["label_tfidf"] = out["_label"].map(label_tfidf)
                    out["label_longest"] = out["_label"].map(label_long)
                    out["label_shortest"] = out["_label"].map(label_short)
                    out["label_representative"] = out["_label"].map(label_repr)
                    out["parent"] = out["_label"].map(lambda x: pick(int(x)))

                    # URL canon
                    if url_col_final and url_col_final in out.columns:
                        canon = {}
                        for p, sub in out.groupby("parent", sort=False):
                            row = sub.loc[sub["keyword"] == p]
                            if row.empty:
                                u = sub[url_col_final].mode().iloc[0] if not sub[url_col_final].isna().all() else None
                            else:
                                u = row[url_col_final].iloc[0]
                            canon[p] = u
                        out["cluster_url"] = out["parent"].map(canon)

                    # Insights
                    n_clusters = out["_label"].nunique()
                    sizes = out.groupby("parent")["keyword"].count().sort_values(ascending=False)
                    insights = {
                        "clusters": int(n_clusters),
                        "avg_size": float(sizes.mean()) if len(sizes)>0 else 0.0,
                        "median_size": float(sizes.median()) if len(sizes)>0 else 0.0
                    }
                    insights.update(meta)
                    return out, insights, X

                outputs = []
                insights_list = []
                if cfg.cluster_separately_by_url and url_col_final:
                    for url_val, sub in df_work.groupby(url_col_final, dropna=False):
                        res, ins, _ = cluster_block(sub)
                        outputs.append(res)
                        ins["url_bucket"] = str(url_val)
                        insights_list.append(ins)
                    df_out = pd.concat(outputs, ignore_index=True)
                    X_main = None
                    st.info("Clustered separately in each URL bucket.")
                else:
                    df_out, ins, X_main = cluster_block(df_work)
                    df_out["url_bucket"] = df_out[url_col_final] if (url_col_final and url_col_final in df_out.columns) else None
                    insights_list = [ins]

                st.subheader("Label display")
                display_label = st.radio("Choose which label column to display as 'Core cluster label'",
                                         ["parent", "label_metric", "label_centroid", "label_tfidf", "label_longest", "label_shortest", "label_representative"],
                                         horizontal=True, index=0)
                df_view = df_out.copy()
                df_view["core_label"] = df_view[display_label]

                # Insights
                total_clusters = df_view["core_label"].nunique()
                cluster_sizes = df_view.groupby("core_label")["keyword"].count().sort_values(ascending=False)
                noise = next((i["noise"] for i in insights_list if "noise" in i), 0)
                extra = []
                if cfg.algorithm == "KMeans":
                    extra.append(f"silhouette: {insights_list[0].get('silhouette', 'n/a'):.3f}")
                if cfg.algorithm == "HDBSCAN":
                    extra.append(f"noise: {noise}")
                st.markdown(f"**Clusters:** {total_clusters} • **Avg size:** {cluster_sizes.mean():.2f} • **Median size:** {cluster_sizes.median():.0f} " + ("• " + " • ".join(extra) if extra else ""))

                # UMAP (optional)
                if X_main is not None:
                    try:
                        umap_xy = compute_umap_2d(X_main)
                        st.subheader("Cluster map (UMAP 2D)")
                        umap_df = pd.DataFrame({"x": umap_xy[:,0], "y": umap_xy[:,1], "label": df_view["core_label"].values})
                        st.scatter_chart(umap_df, x="x", y="y", color="label", height=400)
                    except Exception as e:
                        st.caption(f"(UMAP unavailable or failed: {e})")

                # --- Cluster Manager ---
                st.subheader("🧰 Cluster Manager")

                with st.expander("Rename / merge clusters"):
                    labels_sorted = list(cluster_sizes.index)
                    rename_from = st.selectbox("Cluster to rename/merge", labels_sorted if len(labels_sorted)>0 else ["<none>"])
                    new_name = st.text_input("New name (existing name merges clusters)")
                    if st.button("Apply rename/merge"):
                        if new_name:
                            st.session_state.overrides["rename"][rename_from] = new_name
                            st.success(f"Mapped '{rename_from}' → '{new_name}'")

                with st.expander("Move specific keywords"):
                    dest = st.selectbox("Move to label", list(cluster_sizes.index) if len(cluster_sizes)>0 else ["<none>"])
                    src_label = st.selectbox("Filter keywords by cluster (optional)", ["<All>"] + list(cluster_sizes.index), index=0)
                    if src_label == "<All>":
                        kws_pool = df_view["keyword"].tolist()
                    else:
                        kws_pool = df_view.loc[df_view["core_label"] == src_label, "keyword"].tolist()
                    selected_kws = st.multiselect("Select keywords to move", kws_pool)
                    if st.button("Move selected keywords"):
                        for kw in selected_kws:
                            st.session_state.overrides["move"][kw] = dest
                        st.success(f"Moved {len(selected_kws)} keywords → '{dest}'")

                with st.expander("Must-link / Cannot-link"):
                    col1, col2 = st.columns(2)
                    with col1:
                        ml_a = st.text_input("Must-link A")
                        ml_b = st.text_input("Must-link B")
                        if st.button("Add must-link"):
                            if ml_a and ml_b:
                                st.session_state.overrides["must_link"].append((ml_a, ml_b))
                                st.success(f"Must-link added: {ml_a} ↔ {ml_b}")
                    with col2:
                        cl_a = st.text_input("Cannot-link A")
                        cl_b = st.text_input("Cannot-link B")
                        if st.button("Add cannot-link"):
                            if cl_a and cl_b:
                                st.session_state.overrides["cannot_link"].append((cl_a, cl_b))
                                st.success(f"Cannot-link added: {cl_a} ⟂ {cl_b}")

                # Apply overrides live
                def apply_overrides(df: pd.DataFrame, core_col: str = "core_label") -> pd.DataFrame:
                    ov = st.session_state.overrides
                    out = df.copy()

                    if ov["rename"]:
                        out[core_col] = out[core_col].replace(ov["rename"])

                    if ov["move"]:
                        mask = out["keyword"].isin(ov["move"].keys())
                        out.loc[mask, core_col] = out.loc[mask, "keyword"].map(ov["move"])

                    for a, b in ov["must_link"]:
                        if a in set(out["keyword"]) and b in set(out["keyword"]):
                            la = out.loc[out["keyword"] == a, core_col].iloc[0]
                            lb = out.loc[out["keyword"] == b, core_col].iloc[0]
                            size_a = (out[core_col] == la).sum()
                            size_b = (out[core_col] == lb).sum()
                            target = la if size_a >= size_b else lb
                            out.loc[out["keyword"].isin([a, b]), core_col] = target

                    for a, b in ov["cannot_link"]:
                        if a in set(out["keyword"]) and b in set(out["keyword"]):
                            la = out.loc[out["keyword"] == a, core_col].iloc[0]
                            lb = out.loc[out["keyword"] == b, core_col].iloc[0]
                            if la == lb:
                                new_label = f"{lb} (split)"
                                out.loc[out["keyword"] == b, core_col] = new_label

                    return out

                df_over = apply_overrides(df_view, core_col="core_label")
                st.markdown("**After overrides (preview)**")
                st.dataframe(df_over.head(1000), use_container_width=True)

                # Recompute top cluster sizes after overrides
                cluster_sizes_over = df_over.groupby("core_label")["keyword"].count().sort_values(ascending=False)
                st.markdown("**Top clusters by size (after overrides)**")
                st.dataframe(cluster_sizes_over.head(25).to_frame("count"))

                # Export / import overrides
                ov_json = json.dumps(st.session_state.overrides, ensure_ascii=False, indent=2)
                st.download_button("⬇️ Download overrides.json", data=ov_json, file_name="overrides.json", mime="application/json")

                uploaded_ov = st.file_uploader("Upload overrides.json", type=["json"], key="ov_json")
                if uploaded_ov:
                    try:
                        st.session_state.overrides = json.loads(uploaded_ov.read().decode("utf-8"))
                        st.success("Overrides loaded.")
                    except Exception as e:
                        st.error(f"Failed to load overrides: {e}")

                # Downloads
                cols_show = ["keyword", "core_label", "_label"]
                if url_col_final:
                    cols_show.append(url_col_final)
                cols_show += ["cluster_url", "label_metric", "label_centroid", "label_tfidf"]
                if cfg.parent_metric_col:
                    cols_show.append(cfg.parent_metric_col)
                df_show = df_over[[c for c in cols_show if c in df_over.columns]].copy()
                st.markdown("**Table (post-overrides)**")
                st.dataframe(df_show.head(1000), use_container_width=True)

                csv_final = df_over.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV (post-overrides)", data=csv_final, file_name=f"keywords_clustered_{cfg.algorithm.lower()}_overrides.csv", mime="text/csv")

st.markdown("---")
st.caption("Built by Farky SEO: robust import, flexible mapping, embeddings & classic algorithms, URL-aware controls, Cluster Manager.")
