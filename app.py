
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

# ML / NLP libs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

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

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# PolyFuzz (string matcher)
try:
    from polyfuzz import PolyFuzz
except Exception:
    PolyFuzz = None

st.set_page_config(page_title="Keyword Clustering (SEO)", page_icon="🔎", layout="wide")

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
    keyword_col_candidates: Tuple[str, ...] = ("Keyword", "Query", "Search term")
    url_col_candidates: Tuple[str, ...] = ("URL", "Page", "Page URL", "Landing Page", "Final URL")
    impressions_col_candidates: Tuple[str, ...] = ("Impressions", "Clicks", "Sessions", "Pageviews", "Volume", "Search Volume")
    intent: IntentConfig = field(default_factory=IntentConfig)
    # URL controls
    cluster_separately_by_url: bool = False
    max_keywords_per_url: Optional[int] = None  # per-bucket cap

# -----------------------------
# Utils
# -----------------------------
def detect_encoding_from_bytes(b: bytes) -> str:
    import chardet
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
    # Prefer a metric column if present
    metric_col = _find_first(df, cfg.impressions_col_candidates)
    return {"keyword": key_col, "url": url_col, "metric": metric_col}

def normalize_keywords(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
              .str.lower())

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
        n_clusters=None, affinity="cosine", linkage="average",
        distance_threshold=distance_threshold
    )
    labels = model.fit_predict(X)
    return labels, {}

def algo_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: Optional[int]):
    if hdbscan is None:
        raise ImportError("hdbscan is required. Add to requirements and reinstall.")
    cl = hdbscan.HDBSCAN(
        metric="euclidean",  # embeddings are normalized
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

# PolyFuzz: leave as separate “batch similarity” pathway (string matching)
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
        centroid = X[idx].mean(axis=0, keepdims=True)
        sims = (X[idx] @ centroid.T).ravel()
        out[int(lbl)] = keywords[idx[np.argmax(sims)]]
    return out

def label_by_tfidf_phrase(keywords: List[str], labels: np.ndarray, top_k: int=2) -> Dict[int, str]:
    out = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        texts = [keywords[i] for i in idx]
        vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english").fit(texts)
        tf = vec.transform(texts).sum(axis=0).A1
        terms = np.array(vec.get_feature_names_out())
        top = terms[tf.argsort()[::-1][:top_k]]
        out[int(lbl)] = " ".join(top)
    return out

def label_by_metric(df: pd.DataFrame, labels: np.ndarray, metric_col: Optional[str]) -> Dict[int, str]:
    out = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if metric_col and metric_col in df.columns:
            ser = pd.to_numeric(df.iloc[idx][metric_col], errors="coerce").fillna(0)
            if ser.empty:
                out[int(lbl)] = df.iloc[idx]["keyword"].iloc[0]
            else:
                out[int(lbl)] = df.iloc[idx].iloc[ser.idxmax()]["keyword"]
        else:
            out[int(lbl)] = df.iloc[idx]["keyword"].iloc[0]
    return out

# -----------------------------
# Pipeline helpers
# -----------------------------
def stratified_cap_by_url(df: pd.DataFrame, url_col: Optional[str], cap: Optional[int]) -> pd.DataFrame:
    if not url_col or cap is None:
        return df
    parts = []
    for url, sub in df.groupby(url_col, dropna=False):
        if len(sub) > cap:
            parts.append(sub.sample(n=cap, random_state=42))
        else:
            parts.append(sub)
    return pd.concat(parts, ignore_index=True)

def build_representation(keywords: List[str], cfg: ClusterConfig) -> np.ndarray:
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

# -----------------------------
# UI
# -----------------------------
st.title("🔎 Keyword Clustering for SEO")
st.caption("Semantic clustering with embeddings and classic methods. Map clusters to URLs and pick clear labels.")

with st.sidebar:
    st.header("1) Data")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "tsv", "xlsx", "xls"])
    use_example = st.toggle("Use example data", value=False)

    st.header("2) Representation")
    representation = st.selectbox("Text representation", ["SBERT (MiniLM)", "SBERT (Multilingual)", "TF-IDF"], index=0)

    st.header("3) Algorithm")
    algorithm = st.selectbox("Clustering algorithm", ["HDBSCAN", "KMeans", "Agglomerative", "Graph (Louvain)", "PolyFuzz (string match batches)"], index=0)

    st.header("4) URL-level controls")
    cap_per_url = st.number_input("Max keywords per URL (cap per bucket)", min_value=1, max_value=10000, value=500)
    cluster_by_url = st.toggle("Cluster separately per URL (bucket first)", value=False)

    st.header("5) Labeling")
    label_strategy = st.selectbox("Primary label shown", ["Highest metric", "Centroid keyword", "TF-IDF phrase", "Longest", "Shortest", "Representative"], index=0)
    tfidf_top_k = st.slider("TF-IDF phrase words (n-grams)", 1, 4, 2)
    st.caption("We also compute alternative labels and show them in extra columns.")

    st.header("6) Algo parameters")
    k_val = st.number_input("KMeans: k", min_value=2, max_value=200, value=20)
    dist_thr = st.slider("Agglomerative: distance threshold (cosine)", 0.0, 1.5, 0.25, 0.01)
    hdb_min_size = st.number_input("HDBSCAN: min cluster size", min_value=2, max_value=500, value=5)
    hdb_min_samples = st.number_input("HDBSCAN: min samples (0 = auto)", min_value=0, max_value=500, value=0)
    graph_cut = st.slider("Graph/Louvain: similarity cutoff", 0.0, 1.0, 0.70, 0.01)

    st.header("7) PolyFuzz (if selected)")
    batch_str = st.text_input("Batch similarity thresholds (comma-separated)", "0.6,0.7,0.8")

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
        df_in = read_table_from_upload(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    df_in = None

if df_in is not None:
    st.subheader("Preview input")
    st.dataframe(df_in.head(20), use_container_width=True)

    # Detect columns and override controls
    cols_detect = detect_columns(df_in, ClusterConfig())
    with st.expander("Column detection & overrides", expanded=False):
        st.write("Detected:", cols_detect)
        key_col = st.selectbox("Keyword column", options=list(df_in.columns), index=list(df_in.columns).index(cols_detect["keyword"]) if cols_detect["keyword"] in df_in.columns else 0)
        url_options = ["<None>"] + list(df_in.columns)
        url_default_idx = url_options.index(cols_detect["url"]) if cols_detect["url"] in url_options else 0
        url_col_sel = st.selectbox("URL column (optional)", options=url_options, index=url_default_idx)
        metric_candidates = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        metric_col = st.selectbox("Metric column (for Highest metric labeling)", options=["<Auto>"] + metric_candidates, index=0)

    # Normalize and cap per URL
    df_work = df_in.copy()
    df_work.rename(columns={key_col: "Keyword"}, inplace=True)
    df_work["keyword"] = normalize_keywords(df_work["Keyword"])
    url_col_final = None if url_col_sel == "<None>" else url_col_sel
    if url_col_final and url_col_final in df_work.columns:
        df_work[url_col_final] = df_work[url_col_final].astype(str).str.strip()
    # Apply per-URL cap
    df_work = stratified_cap_by_url(df_work, url_col_final, cap_per_url)

    st.markdown("**Rows after per-URL cap:** {}".format(len(df_work)))

    # Build cfg
    cfg = ClusterConfig(
        representation=representation,
        algorithm="PolyFuzz" if algorithm.startswith("PolyFuzz") else algorithm,
        batch_thresholds=tuple(sorted(set(float(x.strip()) for x in batch_str.split(",") if x.strip()))),
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

    if run_btn:
        with st.spinner("Clustering..."):
            if cfg.algorithm == "PolyFuzz":
                # PolyFuzz batch path (string matching with thresholds)
                matches = polyfuzz_match(df_work["keyword"].tolist(), "TF-IDF")
                # Merge back
                merged = matches.merge(df_work, on="keyword", how="left")
                # Tabs per threshold
                thresholds = sorted(cfg.batch_thresholds)
                tabs = st.tabs([f"PolyFuzz ≥ {t:.2f}" for t in thresholds])
                for tab, thr in zip(tabs, thresholds):
                    with tab:
                        temp = merged[merged["similarity"] >= thr].copy()
                        # Parent naming: representative by default
                        temp["parent_representative"] = temp["cluster_repr"]
                        # Fallback metric parent
                        metric_c = cfg.parent_metric_col or cols_detect["metric"]
                        temp["_lbl"] = temp["cluster_repr"].astype("category").cat.codes
                        labels_metric = label_by_metric(temp.assign(keyword=temp["keyword"]), temp["_lbl"].values, metric_c)
                        temp["label_metric"] = temp["_lbl"].map(labels_metric)
                        # Final chosen label
                        label_col_to_use = "label_metric" if cfg.parent_strategy == "Highest metric" else "parent_representative"
                        temp["parent"] = temp[label_col_to_use]
                        # URL view
                        if url_col_final and url_col_final in temp.columns:
                            # canonical URL = URL of parent row when available, else mode
                            canon = {}
                            for p, sub in temp.groupby("parent", sort=False):
                                row = sub.loc[sub["keyword"] == p]
                                if row.empty:
                                    u = sub[url_col_final].mode().iloc[0] if not sub[url_col_final].isna().all() else None
                                else:
                                    u = row[url_col_final].iloc[0]
                                canon[p] = u
                            temp["cluster_url"] = temp["parent"].map(canon)

                        # Insights
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
                # Optionally cluster per URL bucket
                def cluster_block(block: pd.DataFrame):
                    keywords = block["keyword"].tolist()
                    # Vectorize
                    X = build_representation(keywords, cfg)
                    # Cluster
                    labels, meta = run_algo(X, cfg)
                    # Build frame
                    out = block.copy()
                    out["_label"] = labels
                    # Label strategies
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
                        # representative ~ centroid
                        label_repr[int(lbl)] = label_centroid[int(lbl)]
                    # Choose primary
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
                    return out, insights

                outputs = []
                insights_list = []
                if cfg.cluster_separately_by_url and url_col_final:
                    for url_val, sub in df_work.groupby(url_col_final, dropna=False):
                        res, ins = cluster_block(sub)
                        outputs.append(res)
                        ins["url_bucket"] = str(url_val)
                        insights_list.append(ins)
                    df_out = pd.concat(outputs, ignore_index=True)
                    st.info("Clustered separately in each URL bucket.")
                else:
                    df_out, ins = cluster_block(df_work)
                    df_out["url_bucket"] = df_out[url_col_final] if url_col_final in df_out.columns else None
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

                cols_show = ["keyword", "core_label", "_label"]
                if url_col_final:
                    cols_show.append(url_col_final)
                cols_show += ["cluster_url", "label_metric", "label_centroid", "label_tfidf"]
                if cfg.parent_metric_col:
                    cols_show.append(cfg.parent_metric_col)
                df_show = df_view[ [c for c in cols_show if c in df_view.columns] ].copy()
                st.dataframe(df_show.head(1000), use_container_width=True)

                st.markdown("**Top clusters by size (core label)**")
                st.dataframe(cluster_sizes.head(25).to_frame("count"))

                # Download
                csv = df_view.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv, file_name=f"keywords_clustered_{cfg.algorithm.lower()}.csv", mime="text/csv")

st.markdown("---")
st.caption("Built for SEO: embeddings, multiple algorithms, URL-aware controls, and clear cluster labeling.")
