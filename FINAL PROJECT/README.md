# Drive Wise: Metadata-Aware Automotive RAG Assistant

A production-grade Retrieval-Augmented Generation (RAG) system designed for automotive brochure analysis. Drive Wise leverages metadata-aware filtering, section-aware chunking, and advanced reranking to provide accurate, context-aware responses about vehicle specifications, features, and performance metrics.

## 🚗 Features

- **Section-Aware Chunking**: Intelligently segments PDF brochures by automotive sections (Engine & Performance, Safety, Dimensions, etc.)
- **Metadata Pre-Filtering**: Strict brand/model filtering to eliminate cross-model contamination
- **Cohere Reranking**: Advanced reranking pipeline for optimal context relevance
- **Gemini Generation**: Deterministic LLM responses with strict guardrails
- **Persistent Vector Storage**: Local ChromaDB for offline capability
- **Interactive UI**: Modern Streamlit interface with source attribution

## 🛠️ Tech Stack

- **Vector Database**: ChromaDB (local persistent storage)
- **Embeddings**: Cohere (embed-english-v3.0)
- **Reranking**: Cohere (rerank-english-v3.0)
- **LLM**: Google Gemini 2.5 Flash
- **Framework**: LangChain
- **Frontend**: Streamlit
- **Data Processing**: PyPDF, Polars
- **Language**: Python 3.14+

## 📋 Prerequisites

- Python 3.14 or higher
- Cohere API key
- Google Gemini API key
- 13 Hyundai car brochure PDFs in `./dataset` directory

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG_AST
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
   ```

## 🚀 Setup & Usage

### 1. Initialize Database

```bash
python init_db.py
```

This creates the local ChromaDB instance at `./chroma_db` with the "car_brochures" collection.

### 2. Ingest Data

Place your car brochure PDFs in the `./dataset` directory, then run:

```bash
python ingest.py
```

This will:
- Process all PDFs using PyPDF
- Apply section-aware chunking
- Extract metadata (brand, model, section, page_number)
- Generate Cohere embeddings
- Store vectors in ChromaDB

**Expected Output:**
```
INGESTION SUMMARY
============================================================
Total documents processed: 13
Total chunks ingested: 536
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
RAG_AST/
├── .env                    # API keys (not in git)
├── .gitignore              # Git ignore rules
├── .streamlit/             # Streamlit config (auto-generated)
├── chroma_db/              # Local vector database (not in git)
├── dataset/                # Car brochure PDFs
├── init_db.py              # Database initialization script
├── ingest.py               # Data ingestion pipeline
├── retrieval.py            # Metadata-filtered retrieval with rerarking
├── generation.py           # LLM generation with guardrails
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🔑 API Keys Required

### Cohere API Key
- Used for embeddings and reranking
- Get your key at: https://dashboard.cohere.com/api-keys
- Free tier available

### Gemini API Key
- Used for response generation
- Get your key at: https://makersuite.google.com/app/apikey
- Generous free tier available

## 🎯 Usage Guide

1. **Select Vehicle**: Use the sidebar to select "Hyundai" and your desired model
2. **Ask Questions**: Type natural language queries about:
   - Fuel efficiency and mileage
   - Engine specifications
   - Safety features
   - Dimensions and capacity
   - Interior features
   - Infotainment systems
3. **View Sources**: Expand "🔍 View Official Brochure Verification Sources" to see source attribution

## 🧠 Architecture

### Data Ingestion (`ingest.py`)
- PDF parsing with PyPDF
- Section-aware regex-based chunking
- Metadata extraction from filenames
- Cohere embedding generation
- Batch upsert to ChromaDB

### Retrieval Pipeline (`retrieval.py`)
- Metadata pre-filtering: `{"$and": [{"brand": brand}, {"model": model}]}`
- Vector similarity search (top 12)
- Cohere reranking (top 3)
- Polars telemetry tracking

### Generation Framework (`generation.py`)
- Gemini 2.5 Flash with temperature=0.0
- Strict system prompt with guardrails
- Verbatim fallback for missing information
- Markdown formatting for specs

### Frontend (`app.py`)
- Dynamic brand/model dropdowns
- Reactive chat interface
- Source attribution table
- API status monitoring

## 🛡️ Guardrails

The system enforces strict guardrails:
- Only uses provided brochure context
- Verbatim fallback: "I cannot find that specific specification in the official brochure documentation."
- No cross-model references
- No hallucinations or external knowledge
- Markdown-formatted technical specifications

## 📊 Performance

- **Ingestion Speed**: ~1 second per PDF page
- **Retrieval Latency**: ~500ms (including reranking)
- **Generation Latency**: ~1-2 seconds
- **Total Response Time**: ~2-3 seconds

## 🔧 Troubleshooting

### "No data available" in dropdown
- Ensure `ingest.py` has been run successfully
- Check that PDFs exist in `./dataset` directory
- Verify Cohere API key is valid

### API errors
- Verify API keys in `.env` file
- Check API quota limits
- Ensure internet connectivity for API calls

### Empty ChromaDB
- Delete `./chroma_db` directory
- Re-run `init_db.py`
- Re-run `ingest.py`

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues or questions, please open an issue on the repository.

---

**Built with ❤️ using LangChain, Cohere, and Google Gemini**
