🕌 Islam-Bot (RAG)
An offline intelligent assistant that answers religious queries using Retrieval-Augmented Generation (RAG). It retrieves verified information from a custom dataset of 51,000+ fatwas and summarizes them using a local LLM.

🚀 How It Works
Semantic Search: User queries are converted into vectors using CAMeL-Lab Arabic BERT.

Retrieval: The system finds the Top-5 most relevant fatwas using Cosine Similarity.

Generation: Gemma 3:4b (via Ollama) processes the retrieved text to provide a concise, professional answer.

🛠️ Tech Stack
Language: Python 3.13

Embeddings: CAMeL-Lab BERT (768 dimensions)

LLM Engine: Ollama (Gemma 3:4b / Llama 3.2)

Framework: Flask

Data: 51k+ Scraped fatwas (Bin Baz, IslamQA, IslamWeb)

📂 Project Structure
src/: Core logic (Search & LLM integration).

notebooks/: Data scraping and preprocessing history.

data/: Fatwa database (CSV).

models/: Saved vector embeddings (.pt).

app.py: Flask web interface.

⚙️ Installation & Setup
Install Ollama: Download from ollama.com.

Pull Model: ```bash ollama run gemma3:4b

Install Dependencies:

pip install -r requirements.txt


Run App:

python app.py

## 📊 Dataset & Models
Due to GitHub's file size limits, the dataset and pre-computed embeddings are hosted on Kaggle. 

**Download the required files here:**
👉 [Islamic Fatwa Q&A Dataset on Kaggle](https://www.kaggle.com/datasets/hazemmosalah/50k-islamic-fatwa-q-and-a-dataset-arabic)

**Instructions:**
1. Download `collected.csv` and place it in the `/data` folder.
2. Download `embeddings.pt` and place it in the `/models` folder.
