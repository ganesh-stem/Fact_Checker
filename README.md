# AI-Powered Fact Checker

A system that analyzes news posts or social media statements, extracts key claims, and verifies them against a vector database of verified facts using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **Claim/Entity Detection**: Uses spaCy or a transformer model (dslim/bert-base-NER) to extract key claims and named entities
- **Trusted Fact Base**: CSV-based database of 50 verified facts from government sources
- **Embedding & Retrieval**: FAISS-powered vector similarity search using sentence-transformers
- **AI-Powered Comparison**: Supports OpenAI GPT-4o-mini and Ollama (llama3.2:3b local)
- **Web UI**: Streamlit interface
- **Similarity Threshold**: Filter low-relevance facts using scoring threshold

## Project Structure

```
fact_checker/
├── data/
│   └── verified_facts.csv      # Database of 50 verified facts
├── src/
│   ├── __init__.py
│   ├── claim_extractor.py      # NLP-based claim extraction
│   ├── embeddings.py           # FAISS vector store & retrieval
│   ├── fact_checker.py         # Combines all components (claim extraction, embedding, retrieval, LLM comparison) into a unified fact-checking pipeline.
│   ├── fact_manager.py         # Manages loading and saving verified facts from CSV files.
│   └── llm_comparator.py       # LLM comparison module
├── samples/
│   ├── sample_input.txt        # Sample claims to test
├── app.py                      # Streamlit web interface
├── main.py                     # CLI interface
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables template
└── README.md                   
```

## Quick Start

### 1. Installation


```bash

###########################################
# I have used Python 3.13.7 for this app.
###########################################

cd fact_checker

# Create virtual environment (recommended)
python -m venv venv

# On Windows: 
venv\Scripts\activate

# On Linux / MacOS:
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

AND

python setup.py
```

### 2. Configuration

Edit `.env` to add your OpenAI API key (if using OpenAI):
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Application

#### Streamlit Web UI
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

#### Command Line Interface
```bash
# Single claim
python main.py "The Indian government has announced free electricity to all farmers starting July 2025."

# Interactive mode
python main.py --interactive

```

## Usage Examples

### Input
```
"The Indian government has announced free electricity to all farmers starting July 2025."
```

### Output
```json
{
  "verdict": "False",
  "evidence": [
    "There is no government scheme announced to provide free electricity to all farmers in India as of 2024.",
    "State-specific electricity subsidies exist but there is no national free electricity scheme for farmers."
  ],
  "reasoning": "The claim is contradicted by verified facts stating there is no national free electricity scheme for farmers."
}
```

## Architecture

```
┌─────────────────┐
│   Input Text    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Claim Extractor │ 
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Embedding │  ← sentence-transformers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Search   │  ← Vector similarity search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Comparator  │  ← OpenAI / Ollama
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Verdict + JSON  │
└─────────────────┘
```

## Components

### 1. Claim Extractor (`src/claim_extractor.py`)
- Extracts main claims using dependency parsing
- Identifies named entities (organizations, dates, locations)

### 2. Embedding System (`src/embeddings.py`)
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- FAISS IndexFlatIP for cosine similarity search
- Automatic index persistence

### 3. LLM Comparator (`src/llm_comparator.py`)
- **OpenAI**: GPT-4o-mini via API
- **Ollama**: Local models (Llama 3.2:3b)

### 4. Fact Base (`data/verified_facts.csv`)
- 50 curated facts from PIB India and government sources
- Categories: agriculture, healthcare, economy, defense, space, etc.

## Verdicts

| Verdict | Meaning |
|---------|---------|
| ✅ True | Claim is supported by verified facts |
| ❌ False | Claim is contradicted by verified facts |
| 🤷 Unverifiable | Not enough information to verify or refute |

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `top_k` | `5` | Number of facts to retrieve |
| `similarity_threshold` | `0.554` | Minimum similarity score for facts |

## Requirements

- For Ollama: Install from https://ollama.ai and run `ollama pull llama3.2:3b`

## Troubleshooting

### "No module named 'faiss'"
```bash
pip install faiss-cpu
```

### "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### "OpenAI API error"
Check your API key in `.env` file.

### "Ollama connection refused"
Start Ollama server:
```bash
ollama serve
```
