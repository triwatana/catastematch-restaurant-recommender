import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CA TasteMatch",
    page_icon="🍽️",
    layout="wide"
)

# ─────────────────────────────────────────
# IMPORTS FROM YOUR MODULE FILES
# ─────────────────────────────────────────
from rag_functions import load_rag_model, predict_rag
from content_model import load_content_model, recommend_content
from collab_filtering_functions import train_collab_filter, predict_collab_filter

# ─────────────────────────────────────────
# DATA LOADING  (cached — runs once)
# ─────────────────────────────────────────
@st.cache_data
def load_dataframes():
    restaurants = pd.read_csv("streamlit_app_data/CA_Restaurants_cleaned.csv")
    restaurants["city"]  = restaurants["city"].str.replace(r"\s+", " ", regex=True).str.strip()
    restaurants["price"] = restaurants["price"].fillna(0).astype(int)

    reviews = pd.read_csv("streamlit_app_data/CA_Reviews.csv").iloc[:, 1:]
    users   = pd.read_csv("streamlit_app_data/CA_Users.csv").iloc[:, 1:]
    ca_raw  = pd.read_csv("streamlit_app_data/CA_Restaurants.csv").iloc[:, 1:]
    return restaurants, reviews, users, ca_raw


# ─────────────────────────────────────────
# MODEL LOADING  (cached — runs once)
# ─────────────────────────────────────────
@st.cache_resource
def load_rag(_dummy):
    return load_rag_model(
        chroma_dir="./ca_chroma_db",
        collection_name="ca_restaurants"
    )

@st.cache_resource
def load_cbf(_dummy):
    return load_content_model("CA_Restaurants_cleaned.csv")

@st.cache_resource
def load_cf(_reviews_hash):
    _, reviews, _, _ = load_dataframes()
    model = train_collab_filter(reviews, factors=50, regularization=0.1, iterations=20)
    return model


# ─────────────────────────────────────────
# HYBRID LOGIC
# ─────────────────────────────────────────
def run_hybrid(query, user_idx, n_final,
               rag_collection, rag_embed_model, rag_df,
               cbf_df, cbf_sim_matrix,
               cf_model, cf_reviews_df, cf_restaurants_df):
    """
    Step 1 — RAG : get top-20 candidates from the query.
    Step 2 — CBF : seed with top RAG result, get similar restaurants.
    Step 3 — CF  : get top-20 for the selected user_idx.
    Step 4 — Ranked union: score by how many models agree (1, 2, or 3).
              Tiebreak by RAG match_score.
    """

    # ── Step 1: RAG ──────────────────────────────────────────
    rag_results = predict_rag(
        query, rag_collection, rag_embed_model, rag_df, N=20
    )
    if rag_results.empty:
        return [], "RAG returned no results. Try a different query."

    rag_names = dict(zip(rag_results["name"], rag_results["match_score"]))

    # ── Step 2: CBF ──────────────────────────────────────────
    top_rag_name = rag_results.iloc[0]["name"]
    cbf_raw      = recommend_content(top_rag_name, cbf_df, cbf_sim_matrix, top_n=20)

    # recommend_content returns formatted strings — extract name from first line
    cbf_names = set()
    if isinstance(cbf_raw, list) and cbf_raw[0] != "Restaurant not found":
        for entry in cbf_raw:
            name_line = entry.strip().split("\n")[0].strip()
            if name_line:
                cbf_names.add(name_line)

    # ── Step 3: CF ───────────────────────────────────────────
    try:
        cf_raw    = predict_collab_filter(
            cf_restaurants_df, cf_model, cf_reviews_df, N=20, user_idx=user_idx
        )
        cf_names  = set(cf_raw["name"].tolist()) if not cf_raw.empty else set()
    except Exception:
        cf_names  = set()

    # ── Step 4: Ranked union ─────────────────────────────────
    results = []
    for _, row in rag_results.iterrows():
        name    = row["name"]
        in_cbf  = name in cbf_names
        in_cf   = name in cf_names
        agreement = 1 + int(in_cbf) + int(in_cf)   # 1, 2, or 3

        results.append({
            "name":         name,
            "address":      row.get("address", ""),
            "city":         row.get("city", ""),
            "stars":        row.get("stars", 0.0),
            "price":        row.get("price", 0),
            "categories":   row.get("categories", ""),
            "review_count": row.get("review_count", 0),
            "match_score":  row.get("match_score", 0.0),
            "in_cbf":       in_cbf,
            "in_cf":        in_cf,
            "agreement":    agreement,
        })

    # Sort: agreement desc → match_score desc
    results.sort(key=lambda x: (x["agreement"], x["match_score"]), reverse=True)
    return results[:n_final], None


# ─────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────
AGREEMENT_BADGE = {
    3: ("🥇 All 3 models agree", "#1a7a1a"),
    2: ("🥈 2 models agree",     "#7a6a00"),
    1: ("🔵 RAG only",           "#1a3a7a"),
}

def stars_emoji(stars):
    full = int(stars)
    half = 1 if (stars - full) >= 0.5 else 0
    return "⭐" * full + ("½" if half else "")

def price_str(p):
    try:
        p = int(p)
        return "$" * p if p > 0 else "N/A"
    except:
        return "N/A"

def render_card(r, rank):
    badge_text, badge_color = AGREEMENT_BADGE[r["agreement"]]
    model_tags = ["✅ RAG"]
    if r["in_cbf"]: model_tags.append("✅ CBF")
    if r["in_cf"]:  model_tags.append("✅ CF")

    with st.container():
        col_rank, col_info, col_score = st.columns([0.5, 6, 1.5])

        with col_rank:
            st.markdown(f"### {rank}")

        with col_info:
            st.markdown(
                f"**{r['name']}** &nbsp;&nbsp;"
                f"<span style='background:{badge_color};color:white;"
                f"padding:2px 8px;border-radius:10px;font-size:0.75em'>"
                f"{badge_text}</span>",
                unsafe_allow_html=True
            )
            st.caption(
                f"📍 {r['address']}, {r['city']}  |  "
                f"{stars_emoji(r['stars'])} {r['stars']}/5  |  "
                f"({int(r['review_count'])} reviews)  |  "
                f"Price: {price_str(r['price'])}"
            )
            if r["categories"]:
                st.caption(f"🍽️ {r['categories']}")
            st.caption("  &nbsp;".join(model_tags))

        with col_score:
            st.metric("Match", f"{r['match_score']:.1%}")

        st.divider()


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
def main():
    st.title("🍽️ CA TasteMatch")
    st.caption("Hybrid recommender · RAG + Content-Based + Collaborative Filtering")

    # ── Load data ─────────────────────────────────────────────
    with st.spinner("Loading data..."):
        restaurants, reviews, users, ca_raw = load_dataframes()

    # ── Load models ───────────────────────────────────────────
    with st.spinner("Loading models (first run may take a minute)..."):
        rag_collection, rag_embed = load_rag("rag")
        cbf_df, cbf_sim           = load_cbf("cbf")
        cf_model                  = load_cf(str(len(reviews)))

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        user_idx = st.number_input(
            "CF User Index",
            min_value=0,
            max_value=len(reviews["user_id"].unique()) - 1,
            value=0,
            step=1,
            help="Index of the user for collaborative filtering recommendations."
        )
        n_final = st.slider("Results to show", min_value=3, max_value=20, value=5)

        st.markdown("---")
        st.markdown("**Model legend**")
        st.markdown("🥇 All 3 models agree\n\n🥈 2 models agree\n\n🔵 RAG only")

        st.markdown("---")
        st.markdown("**How it works**")
        st.markdown(
            "1. RAG retrieves top 20 candidates from your query.\n"
            "2. CBF finds restaurants similar to the top RAG result.\n"
            "3. CF recommends restaurants for the selected user.\n"
            "4. Results ranked by how many models agree."
        )

    # ── Query input ───────────────────────────────────────────
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. cheap tacos in santa barbara, romantic thai dinner, vegan cafe...",
    )

    search_clicked = st.button("🔍 Find Restaurants", type="primary")

    if not search_clicked or not query.strip():
        st.info("Enter a query above and click **Find Restaurants** to get started.")
        return

    # ── Run hybrid ────────────────────────────────────────────
    with st.spinner("Running all three models..."):
        results, error = run_hybrid(
            query             = query,
            user_idx          = user_idx,
            n_final           = n_final,
            rag_collection    = rag_collection,
            rag_embed_model   = rag_embed,
            rag_df            = restaurants,
            cbf_df            = cbf_df,
            cbf_sim_matrix    = cbf_sim,
            cf_model          = cf_model,
            cf_reviews_df     = reviews,
            cf_restaurants_df = ca_raw,
        )

    if error:
        st.warning(error)
        return

    if not results:
        st.warning("No results found. Try a different query.")
        return

    # ── Summary bar ───────────────────────────────────────────
    n3 = sum(1 for r in results if r["agreement"] == 3)
    n2 = sum(1 for r in results if r["agreement"] == 2)
    n1 = sum(1 for r in results if r["agreement"] == 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("🥇 All 3 agree", n3)
    c2.metric("🥈 2 agree",     n2)
    c3.metric("🔵 RAG only",    n1)

    st.markdown(f"### Results for: *\"{query}\"*")
    st.markdown("---")

    # ── Render cards ──────────────────────────────────────────
    for i, r in enumerate(results, 1):
        render_card(r, i)


if __name__ == "__main__":
    main()

