import os
import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# --- Path Routing ---
# Since this file lives in 'app/', we go up one level to reach the root project directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

DB_DIR = os.path.join(DATA_DIR, "chroma_db")
CSV_PATH = os.path.join(DATA_DIR, "books_with_emotions.csv")
FALLBACK_COVER = os.path.join(DATA_DIR, "cover-not-found.jpg")

# --- Data & Vector Database Loading ---
if not os.path.exists(DB_DIR):
    raise FileNotFoundError(f"Chroma DB not found at {DB_DIR}. Please run your init_db.py script first.")

print("📖 Loading book metadata...")
books = pd.read_csv(CSV_PATH)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    FALLBACK_COVER,
    books["large_thumbnail"],
)

print("🧠 Loading HuggingFace embedding model and vector database...")
embedding_model = HuggingFaceEmbeddings()
db_books = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)


# --- Core Recommendation Logic ---
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row["description"]) if pd.notna(row["description"]) else ""
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors = str(row["authors"]) if pd.notna(row["authors"]) else "Unknown Author"
        authors_split = authors.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


# --- Gradio UI Layout ---
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommendation System")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    # Whitelist the data directory so Gradio can serve 'cover-not-found.jpg'
    dashboard.launch(allowed_paths=[DATA_DIR])