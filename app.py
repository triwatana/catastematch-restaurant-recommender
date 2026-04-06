import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --------------------------------------------------
# IMPORT MODULES
# --------------------------------------------------
from hybrid_filtering.content_filtering_functions import (
    load_content_model,
    recommend_content
)

from hybrid_filtering.collab_filtering_functions import (
    train_collab_filter,
    predict_collab_filter
)

# RAG (SAFE IMPORT)
try:
    from rag_model.rag_functions import load_rag_model, predict_rag
    RAG_AVAILABLE = True
except:
    RAG_AVAILABLE = False


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="CA TasteMatch", page_icon="🍽️", layout="wide")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    restaurants = pd.read_csv("data/processed/CA_Restaurants_cleaned.csv")
    reviews     = pd.read_csv("data/processed/CA_Reviews.csv").iloc[:, 1:]
    users       = pd.read_csv("data/processed/CA_Users.csv").iloc[:, 1:]
    raw         = pd.read_csv("data/processed/CA_Restaurants.csv").iloc[:, 1:]
    return restaurants, reviews, users, raw


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    restaurants, reviews, users, raw = load_data()

    cbf_df, cbf_sim = load_content_model("data/processed/CA_Restaurants_cleaned.csv")

    cf_model = train_collab_filter(
        reviews,
        factors=50,
        regularization=0.1,
        iterations=20
    )
    
    rag_collection, rag_embed = load_rag_model(
    chroma_dir="./ca_chroma_db",
    collection_name="ca_restaurants"
)

    return cbf_df, cbf_sim, cf_model, restaurants, reviews, raw, rag_collection, rag_embed

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def price_str(p):
    try:
        return "$" * int(p)
    except:
        return "N/A"

def stars_emoji(s):
    return "⭐" * int(s)

# --------------------------------------------------
# HYBRID FUNCTION
# --------------------------------------------------
def hybrid(query, user_idx, n, cbf_df, cbf_sim, cf_model, reviews, raw, rag_collection, rag_embed):

    # --- RAG ---
    if rag_collection:
        try:
            rag_df = predict_rag(query, rag_collection, rag_embed, cbf_df, N=20)
        except:
            rag_df = cbf_df.sample(20)
    else:
        rag_df = cbf_df.sample(20)

    rag_names = set(rag_df["name"])

    # --- CBF ---
    top_name = rag_df.iloc[0]["name"]
    cbf_raw  = recommend_content(top_name, cbf_df, cbf_sim, top_n=20)

    cbf_names = set()
    for r in cbf_raw:
        name = r.split("\n")[0]
        cbf_names.add(name)

    # --- CF ---
    try:
        cf_df = predict_collab_filter(raw, cf_model, reviews, N=20, user_idx=user_idx)
        cf_names = set(cf_df["name"])
    except:
        cf_names = set()

    # --- Combine ---
    results = []

    for _, row in rag_df.iterrows():
        name = row["name"]

        agreement = 1 + int(name in cbf_names) + int(name in cf_names)

        results.append({
            "name": name,
            "stars": row.get("stars", 0),
            "reviews": row.get("review_count", 0),
            "price": row.get("price", 0),
            "categories": row.get("categories", ""),
            "address": f"{row.get('address','')}, {row.get('city','')}, {row.get('state','')}",
            "score": row.get("match_score", 0),
            "agreement": agreement
        })

    results.sort(key=lambda x: (x["agreement"], x["score"]), reverse=True)

    return results[:n]


# --------------------------------------------------
# UI
# --------------------------------------------------
def main():
    st.title("🍽️ CA TasteMatch")
    st.caption("Hybrid Recommender · RAG + Content + Collaborative")

    cbf_df, cbf_sim, cf_model, restaurants, reviews, raw, rag_collection, rag_embed = load_models()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        user_idx = st.number_input("User Index", 0, 1000, 0)
        n_results = st.slider("Results", 3, 15, 5)

    # Input
    query = st.text_input("What are you looking for?")
    run   = st.button("🔍 Find")

    if not run or not query:
        return

    with st.spinner("Running models..."):
        results = hybrid(
            query, user_idx, n_results,
            cbf_df, cbf_sim, cf_model,
            reviews, raw, rag_collection, rag_embed
        )

    # Display
    for i, r in enumerate(results, 1):
        st.markdown(f"""
### {i}. {r['name']}
{stars_emoji(r['stars'])} {r['stars']}/5 ({int(r['reviews'])} reviews)

Cuisine(s): {r['categories']}  
Price: {price_str(r['price'])}  
Address: {r['address']}  
Match Score: {round(r['score'], 3)}
""")
        st.divider()


# --------------------------------------------------
if __name__ == "__main__":
    main()