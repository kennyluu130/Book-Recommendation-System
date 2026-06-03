# Semantic Book Recommendation System

A context-aware Book Recommendation System that leverages Natural Language Processing (NLP) and AI techniques to discover books based on semantic meaning, emotional tone, and genre classification. Instead of relying purely on exact keyword matches or basic metadata filtering, this system indexes vector embeddings of book descriptions to capture deep semantic concepts, allowing users to find books by describing a concept, mood, or storyline.

The system features: data preprocessing and exploratory analysis, advanced NLP modeling (Vector Search, Text Classification, and Sentiment/Emotion Analysis), and a locally deployable interactive **Gradio** web application.

---

## 🧠 Machine Learning & NLP

The project’s notebooks explore and build the core algorithmic foundations of the recommendation engine:

### 1. Data Exploration & Preprocessing (`data-exploration-and-preprocessing.ipynb`)

- **Data Cleansing:** Handles missing data, cleans text attributes, filters records with insufficient information (e.g., descriptions under 25 words), and creates composite textual features like a `tagged_description` combining ISBN and description text.
- **Feature Engineering:** Prepares data for optimal vectorization and saves refined metadata collections (`books_with_emotions.csv`) used by the modeling tasks.

### 2. Vector Search & Semantic Discovery (`vector-search.ipynb`)

- **Embeddings:** Utilizes transformer models via `HuggingFaceEmbeddings` to translate textual descriptions into high-dimensional vector spaces.
- **Vector Database:** Leverages **Chroma DB** to store and query these vector representations efficiently.
- **Semantic Retrieval:** Enables deep conceptual matching. For example, searching for _"a story about forgiveness"_ can successfully surface books dealing with redemption, family healing, and letting go of the past, even if the explicit word "forgiveness" never appears in the title or metadata.

### 3. Text Classification (`text-classification.ipynb`)

- **Genre and Category Tagging:** Trains and evaluates classification algorithms to bucket books into structured, consolidated categories based on raw summary descriptions.
- **Standardization:** Automatically clusters sparse or messy publishing categories into clean, searchable top-level domains (e.g., Fiction, History, Science).

### 4. Sentiment & Emotion Analysis (`sentiment-analysis.ipynb`)

- **Emotional Tone Extraction:** Evaluates text to identify and assign specific emotional tones or moods (such as _Happy, Surprising, Angry, Suspenseful, and Sad_).
- **Sentiment-Based Filtering:** Enriches the metadata with sentiment indicators, enabling the core recommendation engine to map not just _what_ a book is about, but _how it feels_ to read it.

---

## 📂 Project Structure

```text
├── app/
│   └── app.py                        # Main web application file (Gradio UI)
├── data/                             # Data directory
│   ├── books_with_emotions.csv       # Preprocessed book metadata with emotional tags
│   ├── tagged_description.txt        # Combined documents for vector DB ingestion
│   └── chroma_db/                    # Local Chroma vector storage (generated on init)
├── notebooks/
│   ├── data-exploration-and-preprocessing.ipynb
│   ├── vector-search.ipynb
│   ├── text_classification.ipynb
│   └── sentiment_analysis.ipynb
├── init_db.py                        # Database initialization and chunked embedding generation
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation

```

---

## 🛠️ Local Deployment

Follow these instructions to set up the dependencies, build the local vector repository, and launch the interactive dashboard on your machine.

### Prerequisites

Ensure you have **Python 3.8+** installed along with `pip`.

### Step 1: Install Project Dependencies

Install all required data science frameworks, embedding packages, vector store utilities, and user interface components by running:

```bash
pip install -r requirements.txt

```

### Step 2: Initialize the Vector Database

Before launching the application, you must generate the vector indexes. The `init_db.py` script reads the cleaned text data, computes semantic embeddings using Hugging Face transformers, and processes them in memory-safe chunks before saving the indexed database to disk:

```bash
python init_db.py

```

_(Note: A progress indicator will display the progress as it writes to the local `data/chroma_db/` folder)._

### Step 3: Run the Application

With the database initialized, start the web interface using the following command from the root directory:

```bash
python app/app.py

```

---

## 🎮 How to Use the Application

Once the server launches, it will expose a local URL (typically `http://127.0.0.1:7860`). Open it in any browser to access the interface:

1. **Enter Description:** Type a natural language description of what you feel like reading (e.g., _"An astronaut stranded in space"_ or _"A psychological mystery with a twist ending"_).
2. **Apply Filters:** Optionally refine your search using the interactive dropdown menus:

- **Category:** Filter by genres (e.g., Fiction, Science, History, etc.).
- **Emotional Tone:** Select a desired mood (e.g., _Suspenseful, Happy, Sad_).

3. **Get Results:** Click **"Find recommendations"** to instantly query Chroma DB. The UI displays an elegant gallery layout showing corresponding book covers, titles, authors, and brief contextual descriptions.

---

## ⚙️ Technologies Used

- **Frontend UI:** Gradio
- **Vector Store & Indexing:** Chroma DB, LangChain (`langchain-chroma`, `langchain-huggingface`)
- **Embedding Frameworks:** Hugging Face Transformers / Sentence-Transformers
- **Data Processing:** Pandas, NumPy, Scikit-Learn
- **Visualization:** Seaborn, Matplotlib
