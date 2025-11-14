# Aurora QA System

A production-ready question-answering API that answers questions about users by retrieving relevant messages from Aurora's public API and using LLM extraction.

## Overview

This system implements a **retrieval-augmented generation (RAG)** pipeline to answer natural language questions about users based solely on messages retrieved from Aurora's public `/messages` endpoint.

### Architecture

```
User Question
      │
      ▼
 /api/ask (FastAPI)
      │
      ▼
   QA Engine
      ├── Embedding question
      ├── Vector search in FAISS
      ├── Fetch K relevant messages
      ├── LLM fact extraction
      ▼
JSON Answer
```

## Strategy Comparison

This implementation uses **Strategy 2: Retrieval-Based QA**. Here's a comparison of different approaches:

### Strategy 0: Full Dataset to LLM
- **Approach**: Send all messages to LLM for each question
- **Pros**: Simple, no indexing needed
- **Cons**: Expensive, slow, token limits, poor scalability
- **Use Case**: Very small datasets (< 100 messages)

### Strategy 1: Rule-Based Parsing
- **Approach**: Use regex/pattern matching to extract facts
- **Pros**: Fast, no API costs, deterministic
- **Cons**: Brittle, requires domain knowledge, doesn't handle ambiguity
- **Use Case**: Highly structured data with predictable patterns

### Strategy 2: Retrieval-Based Extraction (✅ Chosen)
- **Approach**: Embed messages → FAISS search → Retrieve top-K → LLM extraction
- **Pros**: Scalable, efficient, handles semantic similarity, modern AI approach
- **Cons**: Requires embedding API, initial indexing time
- **Use Case**: Medium to large datasets, semantic queries, production systems

### Strategy 3: Full RAG with Chunking
- **Approach**: Chunk messages → Embed chunks → Multi-stage retrieval → RAG pipeline
- **Pros**: Handles very long documents, advanced retrieval
- **Cons**: Overkill for this use case, more complex
- **Use Case**: Large documents, complex multi-hop reasoning

## Technology Stack

- **FastAPI**: Modern async web framework
- **HuggingFace Inference API**: Free embeddings (`BAAI/bge-small-en-v1.5`)
- **Groq API**: Free LLM extraction (`llama3-8b-instruct`)
- **FAISS**: Vector similarity search (Facebook AI Similarity Search)
- **Pydantic**: Data validation and settings management
- **httpx**: Async HTTP client

## Project Structure

```
aurora-qa/
├── app/
│   ├── api/
│   │   └── ask.py              # POST /api/ask endpoint
│   ├── core/
│   │   ├── config.py           # Environment configuration
│   │   ├── embeddings.py       # HuggingFace embeddings service
│   │   └── llm.py              # LLM extraction service
│   ├── models/
│   │   ├── request.py          # AskRequest model
│   │   ├── response.py         # AskResponse model
│   │   └── message.py          # Message model
│   ├── services/
│   │   ├── ingestion.py        # Fetch messages from Aurora API
│   │   ├── indexing.py         # FAISS index builder
│   │   ├── retrieval.py        # Vector search and retrieval
│   │   └── qa_engine.py        # Main QA orchestrator
│   └── main.py                 # FastAPI app with startup hooks
├── tests/
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- HuggingFace API token (free at https://huggingface.co/settings/tokens)
- Groq API key (free at https://console.groq.com/)
- Access to Aurora's messages API

### Installation

1. **Clone the repository** (or navigate to project directory)

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   Create a `.env` file in the project root:
   ```env
   # HuggingFace (free)
   HF_API_TOKEN=your_huggingface_token_here
   HF_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   
   # Groq (free)
   GROQ_API_KEY=your_groq_key_here
   GROQ_MODEL=llama3-8b-instruct
   
   # Messages API
   MESSAGES_API_URL=https://november7-730026606190.europe-west1.run.app/messages
   
   # Optional
   TOP_K=5
   PORT=8000
   ```

   Or export them:
   ```bash
   export HF_API_TOKEN=your_huggingface_token_here
   export GROQ_API_KEY=your_groq_key_here
   export MESSAGES_API_URL=https://november7-730026606190.europe-west1.run.app/messages
   ```

5. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

   The server will start on `http://localhost:8000`

   On startup, the application will:
   - Fetch all messages from Aurora's API
   - Embed all messages
   - Build the FAISS index
   - Be ready to answer questions

## API Documentation

### POST /api/ask

Answer questions about users based on retrieved messages.

**Request**:
```json
{
  "question": "When is Layla planning her trip to London?"
}
```

**Response**:
```json
{
  "answer": "Layla is planning her trip in March 2024."
}
```

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

**Example using Python**:
```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/ask",
    json={"question": "When is Layla planning her trip to London?"}
)
print(response.json())
```

### GET /health

Health check endpoint to verify the service is running and the index is ready.

**Response**:
```json
{
  "status": "healthy",
  "index_ready": true
}
```

### Interactive API Docs

FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

All configuration is managed through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_API_TOKEN` | HuggingFace API token (required) | - |
| `HF_EMBEDDING_MODEL` | HuggingFace embedding model | `BAAI/bge-small-en-v1.5` |
| `GROQ_API_KEY` | Groq API key (required) | - |
| `GROQ_MODEL` | Groq LLM model | `llama3-8b-instruct` |
| `MESSAGES_API_URL` | Aurora messages API endpoint | `https://november7-730026606190.europe-west1.run.app/messages` |
| `TOP_K` | Number of messages to retrieve | `5` |
| `PORT` | Server port | `8000` |

### Getting Free API Keys

1. **HuggingFace Token**: 
   - Sign up at https://huggingface.co/
   - Go to https://huggingface.co/settings/tokens
   - Create a new token (read access is sufficient)

2. **Groq API Key**:
   - Sign up at https://console.groq.com/
   - Navigate to API Keys section
   - Create a new API key (free tier available)

## Docker Deployment

### Build the image:
```bash
docker build -t aurora-qa .
```

### Run the container:
```bash
docker run -p 8000:8000 \
  -e HF_API_TOKEN=your_hf_token_here \
  -e GROQ_API_KEY=your_groq_key_here \
  -e MESSAGES_API_URL=https://november7-730026606190.europe-west1.run.app/messages \
  aurora-qa
```

### Deploy to Railway

1. Connect your repository to Railway
2. Set environment variables in Railway dashboard:
   - `HF_API_TOKEN` (required)
   - `GROQ_API_KEY` (required)
   - `MESSAGES_API_URL` (optional, has default)
3. Railway will automatically detect the Dockerfile and deploy

The `PORT` environment variable will be automatically set by Railway.

## How It Works

1. **Startup**: On application startup, the system:
   - Fetches all messages from Aurora's API
   - Embeds each message using HuggingFace's `BAAI/bge-small-en-v1.5` (free)
   - Builds a FAISS index for fast similarity search

2. **Question Processing**:
   - User submits a question via `/api/ask`
   - Question is embedded using HuggingFace Inference API
   - FAISS index is queried to find top-K most similar messages
   - Retrieved messages are passed to Groq LLM (`llama3-8b-instruct`) with the question
   - LLM extracts a factual answer based only on the retrieved messages
   - Answer is returned to the user

3. **LLM Prompt**:
   - System prompt instructs the LLM to answer strictly based on provided messages
   - If information is not found, it returns "No information found."
   - Answers are forced to be single-sentence, factual responses

## Data Insights

After inspecting messages from Aurora's API, you may observe:

- **Duplicate user entries**: Some users may appear multiple times with slight variations
- **Inconsistent naming**: User names may have different formats or spellings
- **Timestamp variations**: Different timestamp formats or timezone issues
- **Ambiguous statements**: Messages may contain incomplete or unclear information
- **Missing metadata**: Some messages may lack expected fields

The retrieval-based approach helps handle these issues by:
- Finding semantically similar messages even with different wording
- Aggregating information from multiple relevant messages
- Using LLM to synthesize coherent answers from potentially noisy data

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

The codebase follows:
- Type hints throughout
- Async/await for I/O operations
- Proper error handling and logging
- Clean separation of concerns
- Production-ready patterns

## License

This project is part of Aurora's take-home assignment.

## Free Tier Architecture

This implementation uses **100% free APIs**:
- **HuggingFace Inference API**: Free embeddings with generous rate limits
- **Groq API**: Free LLM inference with fast response times
- **FAISS**: Local CPU-based vector search (no API costs)

No OpenAI costs, no local GPU required, fully cloud-based and scalable.

## Author

Built as a demonstration of modern AI system design, retrieval optimization, clean API construction, and production-quality architecture using free, open-source tools.

