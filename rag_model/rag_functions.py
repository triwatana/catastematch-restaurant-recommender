import re
import ast
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict


# ─────────────────────────────────────────
# QUERY PARSER
# ─────────────────────────────────────────
class QueryParser:
    def __init__(self):
        self.price_keywords = {
            'cheap': [1, 2], 'affordable': [1, 2], 'budget': [1, 2],
            'budget-friendly': [1, 2], 'inexpensive': [1, 2],
            'moderate': [2, 3], 'mid-range': [2, 3],
            'expensive': [3, 4], 'upscale': [3, 4], 'fancy': [3, 4],
            'fine dining': [3, 4], 'luxury': [3, 4],
            'high-end': [3, 4], 'splurge': [3, 4],
        }
        self.rating_keywords = {
            'excellent': 4.5, 'amazing': 4.5, 'outstanding': 4.5,
            'great': 4.0, 'good': 3.5, 'highly rated': 4.0,
            'top rated': 4.5, 'best': 4.5, 'decent': 3.0,
            'okay': 2.5, 'popular': 3.5, 'well-reviewed': 4.0,
        }
        self.cuisine_keywords = [
            'italian', 'chinese', 'japanese', 'mexican', 'indian',
            'thai', 'french', 'american', 'mediterranean', 'greek',
            'korean', 'vietnamese', 'spanish', 'middle eastern',
            'caribbean', 'ethiopian', 'brazilian', 'peruvian',
            'latin', 'salvadoran', 'turkish', 'moroccan',
            'himalayan', 'nepalese', 'indonesian', 'taiwanese',
            'cantonese', 'szechuan', 'pan asian', 'asian fusion',
            'sushi', 'pizza', 'burger', 'seafood', 'steakhouse',
            'ramen', 'noodles', 'hot pot', 'hotpot', 'tacos',
            'sandwiches', 'salad', 'soup', 'tapas', 'dim sum',
            'barbeque', 'bbq', 'wings', 'chicken', 'poke',
            'cheesesteaks', 'falafel', 'kebab', 'wraps',
            'vegetarian', 'vegan', 'gluten-free', 'halal',
            'bakery', 'cafe', 'coffee', 'diner', 'brunch', 'breakfast',
            'dessert', 'ice cream', 'bubble tea', 'juice',
            'sports bar', 'cocktails', 'wine bar', 'beer', 'brewpub',
            'gastropub', 'bar', 'lounge',
            'soul food', 'comfort food', 'buffet', 'food truck',
            'patisserie', 'creperie', 'donuts', 'bagels',
        ]
        self.city_keywords = [
            'santa barbara', 'goleta', 'carpinteria',
            'isla vista', 'montecito', 'summerland',
        ]

    def parse_query(self, query: str) -> Dict:
        query_lower = query.lower()
        filters = {
            'price_filter': None, 'min_rating': None,
            'cuisine_filter': [], 'city_filter': None,
            'cleaned_query': query,
        }
        filters['price_filter']   = self._extract_price(query_lower)
        filters['min_rating']     = self._extract_rating(query_lower)
        filters['cuisine_filter'] = self._extract_cuisines(query_lower)
        filters['city_filter']    = self._extract_city(query_lower)
        filters['cleaned_query']  = self._clean_query(query_lower)
        return filters

    def _extract_price(self, query):
        dollar_match = re.search(r'\$+', query)
        if dollar_match:
            return list(range(1, min(len(dollar_match.group()) + 1, 5)))
        for kw, pr in self.price_keywords.items():
            if kw in query:
                return pr
        under = re.search(r'(?:under|less than|below|max|maximum)\s*\$?(\d+)', query)
        if under:
            mp = int(under.group(1))
            if mp <= 15:   return [1]
            elif mp <= 30: return [1, 2]
            elif mp <= 60: return [1, 2, 3]
            else:          return [1, 2, 3, 4]
        return None

    def _extract_rating(self, query):
        m = re.search(r'(\d+(?:\.\d+)?)\+?\s*(?:star|rating)', query)
        if m: return float(m.group(1))
        for kw, r in self.rating_keywords.items():
            if kw in query: return r
        m2 = re.search(r'(?:above|over|at least|minimum)\s*(\d+(?:\.\d+)?)', query)
        if m2: return float(m2.group(1))
        return None

    def _extract_cuisines(self, query):
        found = []
        for c in self.cuisine_keywords:
            pattern = r'\b' + re.escape(c).replace(r'\ ', r'[\s_-]') + r'\b'
            if re.search(pattern, query):
                found.append(c)
        return found

    def _extract_city(self, query):
        for city in self.city_keywords:
            if city in query:
                return city.title()
        return None

    def _clean_query(self, query):
        original = query
        food_words = set(self.cuisine_keywords)
        for kw in self.rating_keywords:
            if kw.lower() not in food_words:
                query = re.sub(r'\b' + re.escape(kw) + r'\b', '', query,
                               flags=re.IGNORECASE)
        query = re.sub(
            r'(?:under|less than|below|max|maximum|above|over|at least|minimum)\s*\$?\d+',
            '', query)
        query = re.sub(r'\d+\+?\s*(?:star|rating)s?', '', query)
        query = re.sub(r'\$+', '', query)
        for city in self.city_keywords:
            query = re.sub(r'\b' + re.escape(city) + r'\b', '', query,
                           flags=re.IGNORECASE)
        query = ' '.join(query.split()).strip()
        return query if len(query) > 3 else original.strip()


# ─────────────────────────────────────────
# DESCRIPTION BUILDER
# ─────────────────────────────────────────
def parse_dict_field(val):
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, dict) else {}
    except:
        return {}


def create_restaurant_description(row):
    parts = []

    parts.append(f"Restaurant name: {row['name']}")

    if pd.notna(row.get('city')):
        parts.append(f"Located in {row['city']}, California")

    if pd.notna(row.get('categories')):
        cats = [
            c.strip().lower().replace('&', 'and').replace('/', ' ')
            for c in str(row['categories']).split(',')
            if c.strip().lower() not in ('food', 'restaurants')
        ]
        if cats:
            parts.append(f"Cuisine types: {', '.join(cats)}")

    price_labels = {1: "Budget-friendly", 2: "Moderate",
                    3: "Upscale", 4: "Fine dining"}
    price_val = row.get('price')
    if pd.notna(price_val) and int(price_val) in price_labels:
        parts.append(f"Price range: {price_labels[int(price_val)]}")

    try:
        if pd.notna(row.get('stars')):
            parts.append(f"Rating: {float(row['stars'])}/5 stars")
    except (ValueError, TypeError):
        pass

    if pd.notna(row.get('review_count')):
        rc = int(row['review_count'])
        if rc > 500:
            parts.append("Very popular restaurant")
        elif rc > 100:
            parts.append("Popular restaurant")

    if pd.notna(row.get('address')):
        parts.append(f"Location: {row['address']}")

    attrs = parse_dict_field(row.get('attributes', {}))

    ambience = parse_dict_field(attrs.get('Ambience', {}))
    ambience_tags = [k for k, v in ambience.items() if str(v).lower() == 'true']
    if ambience_tags:
        parts.append(f"Ambience: {', '.join(ambience_tags)}")

    noise = str(attrs.get('NoiseLevel', '')).strip("u'\" ")
    if noise and noise not in ('None', ''):
        parts.append(f"Noise level: {noise}")

    good_for = parse_dict_field(attrs.get('GoodForMeal', {}))
    meal_tags = [k for k, v in good_for.items() if str(v).lower() == 'true']
    if meal_tags:
        parts.append(f"Good for: {', '.join(meal_tags)}")

    amenity_map = {
        'OutdoorSeating':          'outdoor seating',
        'RestaurantsDelivery':     'delivery available',
        'RestaurantsTakeOut':      'takeout available',
        'RestaurantsReservations': 'accepts reservations',
        'GoodForKids':             'good for kids',
        'DogsAllowed':             'dog friendly',
        'HappyHour':               'happy hour',
        'HasTV':                   'has TV',
        'WheelchairAccessible':    'wheelchair accessible',
    }
    amenities = [label for key, label in amenity_map.items()
                 if str(attrs.get(key, '')).strip("u'\" ").lower() == 'true']
    if amenities:
        parts.append(f"Features: {', '.join(amenities)}")

    alcohol = str(attrs.get('Alcohol', '')).strip("u'\" ")
    if alcohol and alcohol not in ('none', 'None', ''):
        parts.append(f"Alcohol: {alcohol}")

    wifi = str(attrs.get('WiFi', '')).strip("u'\" ")
    if wifi and wifi not in ('no', 'None', ''):
        parts.append("Has WiFi")

    hours = parse_dict_field(row.get('hours', {}))
    for span in hours.values():
        try:
            close_hr = int(str(span).split('-')[-1].split(':')[0])
            if close_hr >= 21 or close_hr == 0:
                parts.append("Open late")
                break
        except:
            pass

    return '. '.join(parts)


# ─────────────────────────────────────────
# LOAD RAG MODEL
# ─────────────────────────────────────────
def load_rag_model(chroma_dir: str = "./ca_chroma_db",
                   collection_name: str = "ca_restaurants"):
    """
    Load ChromaDB collection and sentence transformer.
    Call once at startup.
    Returns: (collection, embedding_model)
    """
    client          = chromadb.PersistentClient(path=chroma_dir)
    collection      = client.get_collection(collection_name)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, embedding_model


# ─────────────────────────────────────────
# PREDICT / SEARCH
# ─────────────────────────────────────────
def predict_rag(query: str,
                collection,
                embedding_model,
                df: pd.DataFrame,
                N: int = 5) -> pd.DataFrame:
    """
    Search restaurants using the RAG model.

    Args:
        query:           Natural language query string
        collection:      ChromaDB collection from load_rag_model()
        embedding_model: SentenceTransformer from load_rag_model()
        df:              Restaurant dataframe (CA_Restaurants_cleaned.csv)
        N:               Number of results to return

    Returns:
        DataFrame of top N restaurants with match scores
    """
    parser  = QueryParser()
    filters = parser.parse_query(query)

    search_query    = filters['cleaned_query'] or query
    query_embedding = embedding_model.encode(search_query).tolist()

    # Build ChromaDB where clause
    conditions = []
    if filters['price_filter']:
        conditions.append({"price": {"$in": filters['price_filter']}})
    if filters['min_rating']:
        conditions.append({"rating": {"$gte": filters['min_rating']}})
    if filters['city_filter']:
        conditions.append({"city": {"$eq": filters['city_filter']}})

    if len(conditions) > 1:    where_clause = {"$and": conditions}
    elif len(conditions) == 1: where_clause = conditions[0]
    else:                      where_clause = None

    n_results    = N * 4 if filters['cuisine_filter'] else N
    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["distances", "metadatas", "documents"]
    )
    if where_clause:
        query_kwargs["where"] = where_clause

    results = collection.query(**query_kwargs)

    # Post-retrieval cuisine filter
    candidates = list(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0],
        results['documents'][0],
    ))
    if filters['cuisine_filter']:
        filtered = [c for c in candidates
                    if any(f in c[3].lower() for f in filters['cuisine_filter'])]
        candidates = filtered if filtered else candidates

    # Pad to N if cuisine filter left too few
    if len(candidates) < N:
        existing_ids = {c[0] for c in candidates}
        extras = [c for c in list(zip(
            results['ids'][0], results['distances'][0],
            results['metadatas'][0], results['documents'][0]
        )) if c[0] not in existing_ids]
        candidates = (candidates + extras)[:N]

    candidates = candidates[:N]

    # Build output rows
    rows = []
    for rid, dist, meta, doc in candidates:
        try:
            row       = df.iloc[int(rid)]
            cuisine   = [c.strip() for c in str(row.get('categories', '')).split(',')
                         if c.strip().lower() not in ('food', 'restaurants')]
            rev_count = int(row.get('review_count', 0) or 0)
        except Exception:
            cuisine, rev_count = [], 0

        rows.append({
            'name':         meta.get('name', 'Unknown'),
            'address':      meta.get('address', ''),
            'city':         meta.get('city', ''),
            'stars':        float(meta.get('rating', 0.0) or 0.0),
            'price':        int(meta.get('price', 0) or 0),
            'categories':   ", ".join(cuisine),
            'review_count': rev_count,
            'match_score':  round(max(0.0, min(1.0, 1 - (dist / 2.0))), 3),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/processed/CA_Restaurants_cleaned.csv")
    df["city"]  = df["city"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["price"] = df["price"].fillna(0).astype(int)

    collection, embedding_model = load_rag_model()

    results = predict_rag("cheap tacos in santa barbara",
                          collection, embedding_model, df, N=5)

    print(f"\nFound {len(results)} restaurant(s):\n")
    for i, r in results.iterrows():
        print(f"{i+1}. {r['name']}  —  {r['city']}")
        print(f"   {'⭐' * int(r['stars'])} {r['stars']}/5  ({int(r['review_count'])} reviews)")
        print(f"   Cuisine : {r['categories'] or 'N/A'}")
        print(f"   Price   : {'$' * int(r['price']) if r['price'] > 0 else 'N/A'}")
        print(f"   Address : {r['address']}")
        print(f"   Match   : {r['match_score']:.1%}")
        print()