# RAG Chatbot for Ethical Dataset Analysis

A lightweight retrieval-augmented generation (RAG) application for exploring structured datasets through a Persian-language chat interface. The system allows users to ask questions about the available data, retrieve relevant context from the indexed documents, and receive responses grounded in the loaded datasets.

## Overview

This project combines:

- a Streamlit-based chat interface,
- a FAISS vector index for semantic retrieval,
- a simple document ingestion pipeline based on CSV datasets,
- and a prompt-driven answer generation flow for Persian queries.

It is designed to help users interact with dataset content without writing SQL or manually inspecting raw files.

## Project Structure

```text
.
├── app.py                 # Streamlit application entry point
├── build_index.py         # Utility to build the FAISS index locally
├── data_loader.py         # CSV ingestion and text conversion logic
├── rag_engine.py          # Retrieval, indexing, and answer generation logic
├── requirements.txt       # Python dependencies
├── data/                  # CSV datasets used as the knowledge base
├── faiss_index/           # Persisted FAISS index files
├── tests/                 # Regression tests for startup and index behavior
└── README.md              # Project documentation
```

## Features

- Persian-language chat experience
- Retrieval grounded in dataset content
- FAISS-based vector search
- Support for rebuilding the index when needed
- Fallback behavior for local embedding usage in environments without external API access
- Simple test coverage for startup reliability

## Requirements

The project requires Python 3.10+ and the dependencies listed in [requirements.txt](requirements.txt).

### Recommended local environment

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

The application reads data from the CSV files inside the [data](data) directory:

- country_preferences.csv
- demographic_preferences.csv
- moral_machine_responses.csv

These files are converted into text chunks and indexed for retrieval.

## Running the Application

### 1. Build the index

If the FAISS index is missing, build it locally:

```bash
python build_index.py
```

### 2. Start the chat app

```bash
streamlit run app.py
```

## Testing

Regression tests are available for the startup and index-loading flow:

```bash
pytest -q tests/test_startup_fallback.py
```

## Notes

- The app uses a retrieval pipeline that depends on the available index files.
- If no index is present, the system will attempt to build one automatically when possible.
- For deployment environments, the runtime must have the required Python packages installed and accessible.

## License

This project is intended for internal experimentation and educational use.
