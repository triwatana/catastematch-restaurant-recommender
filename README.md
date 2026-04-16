# 🍽️ TasteMatch CA
### Hybrid Restaurant Recommender for Santa Barbara, California

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green?style=flat-square)

> *Find the perfect restaurant — not by keywords, but by what you actually mean.*

---

## 📌 Problem Statement

With hundreds of restaurants in any given area, finding one that truly matches your preferences is a challenge. Keyword search and basic filters fail to capture the *intent* behind queries like:

- *"romantic dinner with outdoor seating"*
- *"cheap tacos in Santa Barbara"*
- *"vegan friendly cafe to work from"*

**TasteMatch CA** solves this by combining three AI/ML models into a unified hybrid pipeline — understanding what you mean, not just what you typed.

---

## 🎯 Objectives

| | Goal |
|---|---|
| 🔍 | Enable natural language restaurant search without structured input |
| 🤖 | Combine semantic search, user preferences, and feature similarity |
| 📊 | Surface high-confidence recommendations via multi-model agreement |
| 🌐 | Deploy as an interactive Streamlit web application |

---

## 📂 Dataset

**Source:** Yelp Academic Dataset — filtered to Santa Barbara area, California

| File | Description | Size |
|---|---|---|
| `yelp_academic_dataset_business.json` | Restaurant details, attributes, hours | 1,036 restaurants |
| `yelp_academic_dataset_review.json` | User reviews and star ratings | ~44,000 reviews |
| `yelp_academic_dataset_user.json` | User profiles | ~40,000 users |

---

## 🤖 Models

<table>
<tr>
<td width="33%" align="center">

### 🔎 RAG
**Semantic Search**

`all-MiniLM-L6-v2` embeddings
stored in ChromaDB with
HNSW cosine similarity.

Rule-based QueryParser extracts
price, rating, city & cuisine
filters before search.

</td>
<td width="33%" align="center">

### 📋 Content-Based
**Feature Similarity**

TF-IDF vectors built from
categories, price & attributes.

Cosine similarity matrix
across all restaurant pairs.

Seeds from top RAG result.

</td>
<td width="33%" align="center">

### 👥 Collaborative
**User Preferences**

ALS matrix factorization
on a 44K-review
user-restaurant matrix.

Recommends based on
preferences of similar users.

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```
                        User Query
                            │
                            ▼
              ┌─────────────────────────┐
              │        RAG Model        │
              │  QueryParser            │
              │    → Filter extraction  │
              │  Sentence Transformer   │
              │    → Query embedding    │
              │  ChromaDB               │
              │    → Top 20 candidates  │
              └────────────┬────────────┘
                           │  candidate pool
               ┌───────────┴───────────┐
               ▼                       ▼
   ┌────────────────────┐  ┌────────────────────┐
   │  Content-Based     │  │  Collaborative     │
   │  Filtering         │  │  Filtering (ALS)   │
   │  (TF-IDF + Cosine) │  │  (User Preferences)│
   └──────────┬─────────┘  └──────────┬─────────┘
              │                       │
              └───────────┬───────────┘
                          ▼
             ┌─────────────────────────┐
             │    Ranked Union         │
             │  🥇 3 models agree      │
             │  🥈 2 models agree      │
             │  🔵 RAG only            │
             └────────────┬────────────┘
                          ▼
                Top-N Recommendations
```

---

## 📁 File Structure

```
TasteMatchCA/
│
├── 📄 app.py                         # Streamlit application
├── 📄 content_model.py               # Content-based filtering
├── 📄 collab_filter_functions.py     # Collaborative filtering
│
├── 📁 ca_chroma_db/                  # Persisted ChromaDB vector store
│
├── 📁 data/
│   └── 📁 processed/
│       ├── CA_Restaurants_cleaned.csv
│       ├── CA_Restaurants.csv
│       ├── CA_Reviews.csv
│       └── CA_Users.csv
│
└── 📁 notebooks/
    ├── content_filtering.ipynb
    └── Collaborative_Filtering.ipynb
└── 📁 rag_model/
    ├── rag_california.ipynb
    ├── rag_functions.py
```

---

## 🚀 How to Run

> To be completed.

---

## 📊 Key Observations

| Finding | Detail |
|---|---|
| ✅ **Perfect Hit Rate** | Hit Rate@5 = 1.0 — at least one relevant result returned for every query |
| 🎯 **Strong Precision** | Precision@5 = 0.783 — ~4 out of 5 results shown are genuinely relevant |
| 📈 **Strong Ranking** | NDCG@5 = 0.833 — relevant results consistently appear near the top |
| 🥇 **Hybrid advantage** | Multi-model agreement acts as a natural quality filter |
| 💬 **Explicit > Intent** | Queries with clear cuisine/price signals achieve Precision@5 = 1.0 |
| 🗄️ **Data richness matters** | Yelp's structured attributes improved embedding quality over the NYC dataset |
| 🕸️ **Sparse matrix** | Most users reviewed only 1–2 restaurants; ALS handles this via matrix factorization |

---

## 👥 Team

| Name | Contribution |
|---|---|
| **Shreyas Prakash** | RAG model |
| **Tri Watanasuparp** | Content-based filtering |
| **Daniel Xiong** |  Collaborative filtering |

---

<p align="center">
  Built with ❤️ using Python · Streamlit · ChromaDB · Sentence Transformers · implicit
</p>
