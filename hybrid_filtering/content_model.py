import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------
# Load and clean data
# -------------------
def load_data(path="data/processed/CA_Restaurants_cleaned.csv"):
    df = pd.read_csv(path)

    # Fill missing values
    df["name"] = df["name"].fillna("")
    df["categories"] = df["categories"].fillna("")
    df["price"] = df["price"].fillna(0).astype(str)
    df["attributes"] = df["attributes"].fillna("")
    df["stars"] = df["stars"].fillna(0)

    return df



# -------------------
# Feature engineering 
# -------------------
def create_features(df):
    # Clean categories
    df["categories_clean"] = df["categories"].str.replace(",", " ", regex=False)

    # Combine features
    df["features"] = (
        df["categories_clean"] + " " +
        "price_" + df["price"] + " " +
        df["attributes"]
    )

    return df


# ------------------
# Build TF-IDF model
# ------------------
def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(df["features"])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


# ------------------
# Stars emoji helper
# ------------------
def get_star_emoji(stars):
    # convert rating → emojis
    if stars >= 4.5:
        return "⭐⭐⭐⭐⭐"
    elif stars >= 4.0:
        return "⭐⭐⭐⭐"
    elif stars >= 3.0:
        return "⭐⭐⭐"
    elif stars >= 2.0:
        return "⭐⭐"
    else:
        return "⭐"


def format_stars(stars, review_count):
    return f"{round(stars,1)}/5 ({int(review_count)} reviews)"


# ------------------------------
# Convert price to dollar helper
# ------------------------------
def price_to_dollar(price):
    try:
        p = int(float(price))
    except:
        return "N/A"

    if p <= 0:
        return "N/A"

    return "$" * p

# -------------------
# Clean cuisine helper
# -------------------
def clean_cuisine(categories):
    if not isinstance(categories, str):
        return ""

    # split → remove "Restaurants" → strip spaces
    items = [c.strip() for c in categories.split(",") if c.strip() != "Restaurants"]

    return ", ".join(items)

# ------------------
# Recommend function
# ------------------
def recommend_content(restaurant_name, df, similarity_matrix, top_n=5):
    if restaurant_name not in df["name"].values:
        return ["Restaurant not found"]

    idx = df[df["name"] == restaurant_name].index[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for i in scores[1:top_n + 1]:
        row = df.iloc[i[0]]

        emoji = get_star_emoji(row["stars"])
        star_text = format_stars(row["stars"], row["review_count"])
        rating_display = f"{emoji} {star_text}"

        price = price_to_dollar(row["price"])

        # Clean full address
        full_address = f"{row['address']}, {row['city']}, {row['state']}"

        # Clean cuisine
        cuisine = clean_cuisine(row["categories"])

        formatted = (
            f"{row['name']}\n"
            f"{rating_display}\n"
            f"Cuisine(s): {cuisine}\n"
            f"Price: {price}\n"
            f"Address: {full_address}\n"
            f"Match Score: {round(float(i[1]), 3)}\n\n"
        )

        recommendations.append(formatted)

    return recommendations


# --------------------------------------------------
# Full pipeline loader (used in Streamlit)
# --------------------------------------------------
def load_content_model(path="data/processed/CA_Restaurants_cleaned.csv"):
    df = load_data(path)
    df = create_features(df)
    similarity_matrix = build_model(df)

    return df, similarity_matrix


# --------------------------------------------------
# Testing the content-based filtering pipeline
# --------------------------------------------------
if __name__ == "__main__":
    df, sim_matrix = load_content_model()

    test_name = df["name"].iloc[3]

    print("\nRecommendations for:", test_name, "\n")

    recs = recommend_content(test_name, df, sim_matrix)

    for r in recs:
        print(r)