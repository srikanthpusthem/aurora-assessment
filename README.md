# Aurora QA System - Backend Documentation

A production-ready question-answering API that answers questions about users by retrieving relevant messages from Aurora's public API and using LLM extraction. This system implements a **retrieval-augmented generation (RAG)** pipeline with a 100% free architecture.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Complete Workflow](#complete-workflow)
6. [Setup & Installation](#setup--installation)
7. [API Documentation](#api-documentation)
8. [Configuration Reference](#configuration-reference)
9. [Logging System](#logging-system)
10. [External API Integration](#external-api-integration)
11. [Performance & Optimization](#performance--optimization)
12. [Deployment](#deployment)
13. [Troubleshooting](#troubleshooting)
14. [Development Guide](#development-guide)

---

## Quick Start

Get the system running in 5 minutes:

```bash
# 1. Clone and navigate to project
cd Aurora_Assessment

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with your API keys
cat > .env << EOF
HF_API_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_key_here
MESSAGES_API_URL=https://november7-730026606190.europe-west1.run.app/messages
EOF

# 5. Run the application
uvicorn app.main:app --reload
```

The server will start on `http://localhost:8000`. On startup, it will:
- Fetch all messages from Aurora's API
- Embed all messages using HuggingFace
- Build the FAISS index
- Be ready to answer questions

**Test the API:**
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

---

## Overview

### System Purpose

The Aurora QA System answers natural language questions about users by:
1. Retrieving relevant messages from Aurora's public API
2. Building a vector index using FAISS for efficient similarity search
3. Using semantic search to find the most relevant messages for a question
4. Extracting factual answers using a Large Language Model (LLM)

### Key Features

- **100% Free Architecture**: Uses HuggingFace Inference API and Groq API (no costs)
- **Semantic Search**: FAISS-based vector similarity search for accurate retrieval
- **Production-Ready**: Comprehensive error handling, logging, and monitoring
- **Fast Response Times**: Optimized pipeline with async operations
- **Scalable**: Handles small to medium datasets efficiently
- **Well-Documented**: Complete workflow and implementation documentation

### Technology Stack

| Component | Technology | Version/Purpose |
|-----------|-----------|----------------|
| **Framework** | FastAPI | 0.104.1 - Modern async web framework |
| **Vector Search** | FAISS | >=1.9.0 - Facebook AI Similarity Search |
| **Embeddings** | HuggingFace Inference API | `BAAI/bge-small-en-v1.5` - Free embeddings |
| **LLM** | Groq API | `llama3-8b-instruct` - Free LLM inference |
| **HTTP Client** | httpx | 0.25.2 - Async HTTP requests |
| **Validation** | Pydantic | 2.5.0 - Data validation and settings |
| **Configuration** | pydantic-settings | 2.1.0 - Environment-based config |
| **Python** | Python | 3.11+ - Runtime |

### Strategy Comparison

This implementation uses **Strategy 2: Retrieval-Based QA**. Here's a comparison:

| Strategy | Approach | Pros | Cons | Use Case |
|----------|----------|------|------|----------|
| **0: Full Dataset to LLM** | Send all messages to LLM | Simple, no indexing | Expensive, slow, token limits | < 100 messages |
| **1: Rule-Based Parsing** | Regex/pattern matching | Fast, no API costs | Brittle, requires domain knowledge | Highly structured data |
| **2: Retrieval-Based (✅)** | Embed → FAISS → Retrieve → LLM | Scalable, efficient, semantic | Requires embedding API | Medium to large datasets |
| **3: Full RAG with Chunking** | Chunk → Multi-stage retrieval | Handles long documents | Overkill, complex | Large documents |

---

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (Frontend)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /api/ask
                             │ { "question": "..." }
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (Backend)                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API Layer (app/api/ask.py)                              │  │
│  │  • Request validation                                    │  │
│  │  • Error handling                                        │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  QA Engine (app/services/qa_engine.py)                   │  │
│  │  • Orchestrates retrieval + extraction                  │  │
│  └────────────┬───────────────────────┬───────────────────────┘  │
│               │                       │                          │
│               ▼                       ▼                          │
│  ┌──────────────────────┐  ┌──────────────────────────────┐  │
│  │  Retrieval Service    │  │  LLM Service                  │  │
│  │  (retrieval.py)       │  │  (core/llm.py)               │  │
│  │  • Embed question     │  │  • Build prompt              │  │
│  │  • Search FAISS       │  │  • Call Groq API            │  │
│  └──────────┬───────────┘  └───────────────┬──────────────┘  │
│             │                                │                 │
│             ▼                                │                 │
│  ┌──────────────────────┐                   │                 │
│  │  Embeddings Service  │                   │                 │
│  │  (core/embeddings.py)│                   │                 │
│  │  • HuggingFace API   │                   │                 │
│  └──────────┬───────────┘                   │                 │
│             │                                │                 │
│             ▼                                │                 │
│  ┌──────────────────────┐                   │                 │
│  │  FAISS Index         │                   │                 │
│  │  (services/indexing) │                   │                 │
│  │  • Vector search     │                   │                 │
│  └──────────────────────┘                   │                 │
│                                             │                 │
│                       ┌─────────────────────┘                 │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Response: { "answer": "..." }                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ (Startup only)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                             │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Aurora API      │  │  HuggingFace API │  │  Groq API    │ │
│  │  /messages       │  │  /hf-inference   │  │  /openai/v1  │ │
│  │  (Ingestion)     │  │  (Embeddings)    │  │  (LLM)       │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Component Layers

1. **API Layer** (`app/api/`): FastAPI routes, request/response validation
2. **Service Layer** (`app/services/`): Business logic orchestration (QA engine, retrieval, indexing)
3. **Core Layer** (`app/core/`): External API integrations (embeddings, LLM, config)
4. **Model Layer** (`app/models/`): Pydantic models for data validation
5. **Configuration Layer** (`app/core/config.py`): Environment-based settings management

### Project Structure

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
├── logs/                       # Log files (auto-created)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env                        # Environment variables (not in repo)
├── start.sh                    # Startup script (macOS)
├── README.md                   # This file
├── BACKEND_FLOW.md            # Detailed flow documentation
└── LOGGING.md                  # Logging guide
```

---

## Implementation Details

### Component Breakdown

#### 1. API Layer

**`app/main.py`** - FastAPI application entry point
- Creates FastAPI app instance
- Configures CORS middleware
- Registers API routes
- Defines startup event handler for index building
- Provides health check endpoint (`/health`)
- Configures logging system

**`app/api/ask.py`** - Question answering endpoint
- Handles `POST /api/ask` requests
- Validates `AskRequest` using Pydantic
- Calls QA engine to process questions
- Returns `AskResponse` with answer
- Comprehensive error handling and logging

#### 2. Service Layer

**`app/services/qa_engine.py`** - QA orchestrator
- High-level coordination of retrieval and extraction
- Calls `retrieve_relevant_messages()` for semantic search
- Calls `extract_answer()` for LLM-based answer extraction
- Returns safe fallback on errors

**`app/services/retrieval.py`** - Message retrieval
- Embeds incoming questions using HuggingFace API
- Searches FAISS index for top-K similar messages
- Returns list of relevant `Message` objects

**`app/services/indexing.py`** - FAISS index management
- `VectorIndex` class for index lifecycle
- `build_index()`: Fetches messages, embeds them, builds FAISS index
- `search()`: Performs vector similarity search
- `is_ready()`: Checks index readiness

**`app/services/ingestion.py`** - Message fetching
- Fetches messages from Aurora's public API
- Handles JSON parsing and response format variations
- Converts API responses to `Message` Pydantic models

#### 3. Core Layer

**`app/core/embeddings.py`** - Text embedding service
- `embed_text()`: Embeds single text (questions)
- `embed_messages()`: Batch embeds messages (startup)
- Integrates with HuggingFace Inference API
- Handles retries for 503 errors (model loading)
- Comprehensive logging of API calls and responses

**`app/core/llm.py`** - LLM answer extraction
- `build_user_prompt()`: Constructs prompt with question and messages
- `extract_answer()`: Calls Groq API for answer extraction
- Formats answers (ensures sentence punctuation)
- Returns safe fallback on errors

**`app/core/config.py`** - Configuration management
- `Settings` class using Pydantic BaseSettings
- Loads from `.env` file and environment variables
- Validates types and provides defaults
- Centralized configuration access

#### 4. Model Layer

**`app/models/message.py`** - Message data model
- `Message` class matching Aurora API response
- Fields: `id`, `user_id`, `message`, `user_name`, `timestamp`
- `text` property for backward compatibility

**`app/models/request.py`** - API request model
- `AskRequest` with `question` field (min_length=1)

**`app/models/response.py`** - API response model
- `AskResponse` with `answer` field

### Design Patterns

- **Dependency Injection**: Services receive dependencies via function parameters
- **Singleton Pattern**: Global `VectorIndex` instance via `get_index()`
- **Repository Pattern**: `VectorIndex` abstracts FAISS operations
- **Strategy Pattern**: Pluggable embedding and LLM providers
- **Factory Pattern**: `get_index()` creates/returns index instance

### Code Organization Principles

- **Separation of Concerns**: Each module has a single responsibility
- **Async/Await**: All I/O operations are asynchronous
- **Type Hints**: Full type annotations throughout
- **Error Handling**: Defensive programming with fallbacks
- **Logging**: Comprehensive logging at each step
- **Configuration**: Environment-based configuration management

---

## Complete Workflow

### Startup Workflow

The application follows a specific startup sequence to build the FAISS index before accepting requests.

```
Application Start
       │
       ▼
┌──────────────────────┐
│  FastAPI App Init    │
│  • Create app        │
│  • Configure CORS     │
│  • Register routes    │
│  • Setup logging      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  @app.on_event       │
│  ("startup")         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  get_index()         │
│  • Create VectorIndex │
│    instance          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  index.build_index() │
└──────────┬───────────┘
           │
           ├─────────────────────────────────────┐
           │                                     │
           ▼                                     ▼
┌──────────────────────┐            ┌──────────────────────┐
│  fetch_messages()    │            │  embed_messages()     │
│  (ingestion.py)      │            │  (embeddings.py)      │
│                      │            │                       │
│  1. HTTP GET to      │            │  1. Extract texts     │
│     Aurora API       │            │  2. Batch POST to      │
│  2. Parse JSON       │            │     HuggingFace API   │
│  3. Create Message   │            │  3. Convert to numpy  │
│     objects          │            │     array             │
└──────────┬───────────┘            └──────────┬───────────┘
           │                                     │
           └──────────────┬──────────────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │  FAISS Index Build   │
                │  (indexing.py)       │
                │                      │
                │  1. Get embedding    │
                │     dimension        │
                │  2. Create           │
                │     IndexFlatL2      │
                │  3. Add embeddings   │
                │  4. Mark as ready    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Index Ready          │
                │  • _is_built = True   │
                │  • Accept requests    │
                └───────────────────────┘
```

**Detailed Startup Steps:**

1. **Application Initialization** (`app/main.py`)
   - FastAPI app creation with metadata
   - CORS middleware configuration
   - Route registration
   - Logging system setup (console + file)

2. **Startup Event Handler**
   - `@app.on_event("startup")` triggers index building
   - Blocking: Application won't accept requests until index is built
   - Fail-fast: Raises exception if index building fails

3. **Message Ingestion** (`app/services/ingestion.py`)
   - HTTP GET request to Aurora API
   - Parses JSON response (`{total, items: [...]}` format)
   - Converts to `Message` Pydantic models
   - Handles 307 redirects automatically

4. **Message Embedding** (`app/core/embeddings.py`)
   - Extracts text content from all messages
   - Batch POST request to HuggingFace Inference API
   - Converts response to numpy array `(N, 384)`
   - Retry logic for 503 errors (model loading)

5. **FAISS Index Construction** (`app/services/indexing.py`)
   - Gets embedding dimension (384 for `bge-small-en-v1.5`)
   - Creates `IndexFlatL2` (exact L2 distance search)
   - Adds all embeddings to index
   - Marks index as ready (`_is_built = True`)

**Startup Timing:**
- Message Fetching: 1-3 seconds
- Embedding Generation: 5-15 seconds (depends on message count)
- Index Construction: < 1 second
- **Total**: ~10-20 seconds for typical datasets

### Request Processing Workflow

When a client sends a question to `/api/ask`, the system processes it through a multi-stage pipeline.

```
Client Request
    │
    │ POST /api/ask
    │ { "question": "Where is Sophia going?" }
    ▼
┌─────────────────────────────────────────┐
│  API Layer (app/api/ask.py)             │
│  • Validate request (Pydantic)          │
│  • Strip whitespace                     │
│  • Check non-empty                      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  QA Engine (app/services/qa_engine.py) │
│  answer_question(question)               │
└──────────────┬───────────────────────────┘
               │
               ├──────────────────────────────┐
               │                              │
               ▼                              │
┌─────────────────────────────────────────┐ │
│  Retrieval Service                       │ │
│  retrieve_relevant_messages(question)    │ │
│                                          │ │
│  1. Embed question                       │ │
│     embed_text(question)                 │ │
│     └─> HuggingFace API                  │ │
│                                          │ │
│  2. Search FAISS index                   │ │
│     index.search(query_embedding, k=5)   │ │
│                                          │ │
│  3. Return top-K messages                │ │
└──────────────┬───────────────────────────┘ │
               │                              │
               └──────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────┐
│  LLM Service (app/core/llm.py)          │
│  extract_answer(question, messages)      │
│                                          │
│  1. Build prompt                         │
│     build_user_prompt(question, msgs)   │
│                                          │
│  2. Call Groq API                        │
│     POST /openai/v1/chat/completions     │
│                                          │
│  3. Extract answer from response         │
│                                          │
│  4. Format answer (ensure sentence)      │
└──────────────┬───────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Response                                │
│  { "answer": "Sophia is going to Paris." }│
└─────────────────────────────────────────┘
```

**Detailed Request Processing Steps:**

1. **API Request Validation** (`app/api/ask.py`)
   - Pydantic validates `AskRequest` schema
   - Strips whitespace and checks non-empty
   - Returns 400 Bad Request for invalid input

2. **QA Engine Orchestration** (`app/services/qa_engine.py`)
   - Calls `retrieve_relevant_messages()` for semantic search
   - If no messages, returns "No information found."
   - Calls `extract_answer()` for LLM extraction
   - Returns safe fallback on errors

3. **Question Embedding** (`app/core/embeddings.py`)
   - Single text embedding request to HuggingFace API
   - Returns numpy array of shape `(384,)`
   - Handles both single-item and nested list responses

4. **FAISS Vector Search** (`app/services/indexing.py`)
   - Reshapes query to `(1, 384)` for FAISS API
   - Calls `index.search()` which returns distances and indices
   - Maps indices back to `Message` objects
   - Returns list of `(Message, distance)` tuples sorted by relevance

5. **LLM Answer Extraction** (`app/core/llm.py`)
   - Builds prompt with system instruction and user question + messages
   - Calls Groq API with OpenAI-compatible format
   - Extracts answer from response
   - Formats answer (ensures sentence punctuation)

**Request Processing Timing:**
- Question Embedding: 0.5-2 seconds
- FAISS Search: < 0.01 seconds
- LLM Extraction: 1-3 seconds
- **Total**: ~2-5 seconds per request

### Data Flow

**Startup Data Flow:**
```
Aurora API → JSON Response → Message Objects → Text Extraction
    ↓
HuggingFace API → Embeddings Array (N, 384) → FAISS Index
    ↓
Index Ready (in-memory)
```

**Request Data Flow:**
```
Question → Embedding (384,) → FAISS Search → Top-K Messages
    ↓
Question + Messages → Prompt → Groq API → Answer
    ↓
Formatted Answer → JSON Response
```

---

## Setup & Installation

### Prerequisites

- **Python**: 3.11 or higher
- **HuggingFace API Token**: Free at https://huggingface.co/settings/tokens
- **Groq API Key**: Free at https://console.groq.com/
- **Access**: Aurora's messages API (public endpoint)

### Installation Steps

1. **Clone the repository** (or navigate to project directory)
   ```bash
   cd Aurora_Assessment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # HuggingFace Configuration (Required)
   HF_API_TOKEN=your_huggingface_token_here
   HF_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   
   # Groq Configuration (Required)
   GROQ_API_KEY=your_groq_key_here
   GROQ_MODEL=llama3-8b-instruct
   
   # Messages API Configuration
   MESSAGES_API_URL=https://november7-730026606190.europe-west1.run.app/messages
   
   # Retrieval Configuration
   TOP_K=5
   
   # Server Configuration
   PORT=8000
   
   # Logging Configuration
   LOG_LEVEL=INFO
   LOG_FILE=logs/aurora_qa.log
   LOG_MAX_BYTES=10485760
   LOG_BACKUP_COUNT=5
   ```

   Or export them:
   ```bash
   export HF_API_TOKEN=your_huggingface_token_here
   export GROQ_API_KEY=your_groq_key_here
   ```

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

   The server will start on `http://localhost:8000`

   On startup, the application will:
   - Fetch all messages from Aurora's API
   - Embed all messages using HuggingFace
   - Build the FAISS index
   - Be ready to answer questions

### Verification

1. **Check health endpoint:**
   ```bash
   curl http://localhost:8000/health
   ```
   
   Expected response:
   ```json
   {
     "status": "healthy",
     "index_ready": true
   }
   ```

2. **Test question endpoint:**
   ```bash
   curl -X POST "http://localhost:8000/api/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "When is Layla planning her trip to London?"}'
   ```

3. **View interactive API docs:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Getting Free API Keys

1. **HuggingFace Token**: 
   - Sign up at https://huggingface.co/
   - Go to https://huggingface.co/settings/tokens
   - Create a new token (read access is sufficient)

2. **Groq API Key**:
   - Sign up at https://console.groq.com/
   - Navigate to API Keys section
   - Create a new API key (free tier available)

---

## API Documentation

### POST /api/ask

Answer questions about users based on retrieved messages.

**Endpoint:** `POST /api/ask`

**Request Body:**
```json
{
  "question": "When is Layla planning her trip to London?"
}
```

**Response:**
```json
{
  "answer": "Layla is planning her trip in March 2024."
}
```

**Status Codes:**
- `200 OK`: Successfully processed question
- `400 Bad Request`: Invalid or empty question
- `500 Internal Server Error`: Processing failure

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

**Example using Python:**
```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/ask",
    json={"question": "When is Layla planning her trip to London?"}
)
print(response.json())
# {"answer": "Layla is planning her trip in March 2024."}
```

**Example using JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/api/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "When is Layla planning her trip to London?"
  })
});
const data = await response.json();
console.log(data.answer);
```

### GET /health

Health check endpoint to verify the service is running and the index is ready.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "index_ready": true
}
```

**Status Codes:**
- `200 OK`: Service is running

**Use Cases:**
- Verify application is running
- Check if index is ready
- Monitor application health
- Load balancer health checks

### Interactive API Docs

FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Configuration Reference

All configuration is managed through environment variables loaded from a `.env` file.

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_API_TOKEN` | HuggingFace API token for embeddings | `hf_xxxxxxxxxxxxx` |
| `GROQ_API_KEY` | Groq API key for LLM | `gsk_xxxxxxxxxxxxx` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MESSAGES_API_URL` | `https://november7-730026606190.europe-west1.run.app/messages` | Aurora API endpoint |
| `HF_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model name |
| `GROQ_MODEL` | `llama3-8b-instruct` | Groq LLM model name |
| `TOP_K` | `5` | Number of messages to retrieve |
| `PORT` | `8000` | Server port number |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE` | `logs/aurora_qa.log` | Log file path (empty = console only) |
| `LOG_MAX_BYTES` | `10485760` | Max log file size (10MB) |
| `LOG_BACKUP_COUNT` | `5` | Number of backup log files to keep |

### Configuration Loading

Configuration is loaded using `pydantic-settings`:

1. **Load from `.env` file**: Automatically reads `.env` in project root
2. **Environment variable override**: Environment variables take precedence
3. **Type validation**: Pydantic validates types (str, int, etc.)
4. **Case insensitive**: Variable names are case-insensitive

### Accessing Configuration

Configuration is accessed via the global `settings` object:

```python
from app.core.config import settings

# Access configuration
api_token = settings.hf_api_token
model_name = settings.hf_embedding_model
top_k = settings.top_k
```

---

## Logging System

The Aurora QA System includes comprehensive logging that outputs to both the console and a log file for easy debugging and learning.

### Log File Location

By default, logs are written to: `logs/aurora_qa.log`

The `logs/` directory is automatically created if it doesn't exist.

### Configuration

You can configure logging in your `.env` file:

```env
# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Log file path (empty = console only)
LOG_FILE=logs/aurora_qa.log

# Log rotation settings (optional)
LOG_MAX_BYTES=10485760  # 10MB per file
LOG_BACKUP_COUNT=5      # Keep 5 backup files
```

### Viewing Logs

**Real-time Console Output:**
Logs appear in your terminal when running the application.

**View Log File:**
```bash
# View logs in real-time
tail -f logs/aurora_qa.log

# View last 50 lines
tail -n 50 logs/aurora_qa.log

# View entire log file
cat logs/aurora_qa.log

# View with pagination
less logs/aurora_qa.log
```

**Filter Logs by Component:**
```bash
# View only QA engine logs
grep "\[QA\]" logs/aurora_qa.log

# View only retrieval logs
grep "\[RETRIEVAL\]" logs/aurora_qa.log

# View only LLM logs
grep "\[LLM\]" logs/aurora_qa.log

# View only embedding logs
grep "\[EMBEDDING\]" logs/aurora_qa.log

# View only startup logs
grep "\[STARTUP\]" logs/aurora_qa.log

# View only request/response logs
grep "\[REQUEST\]\|\[RESPONSE\]" logs/aurora_qa.log
```

**Filter by Log Level:**
```bash
# View only errors
grep "ERROR" logs/aurora_qa.log

# View warnings and errors
grep -E "WARNING|ERROR" logs/aurora_qa.log

# View debug messages (if LOG_LEVEL=DEBUG)
grep "DEBUG" logs/aurora_qa.log
```

### Log Format

Each log entry follows this format:

```
[YYYY-MM-DD HH:MM:SS] LEVEL logger_name - [TAG] message
```

**Example:**
```
[2024-01-15 10:30:45] INFO app.main - [STARTUP] Starting Aurora QA System...
[2024-01-15 10:31:00] INFO app.api.ask - [REQUEST] POST /api/ask - Question: "Where is Sophia going?"
[2024-01-15 10:31:01] INFO app.services.qa_engine - [QA] Processing question: "Where is Sophia going?"
```

### Log Tags

The system uses consistent tags to identify different operations:

- `[STARTUP]` - Application startup and initialization
- `[STEP X/4]` - Startup steps (1=ingestion, 2=embedding, 3=indexing, 4=ready)
- `[REQUEST]` - Incoming API requests
- `[RESPONSE]` - API responses
- `[QA]` - QA engine orchestration
- `[RETRIEVAL]` - Message retrieval operations
- `[EMBEDDING]` - Text embedding operations
- `[LLM]` - LLM extraction operations
- `[SEARCH]` - FAISS index search operations

### Log Rotation

Logs are automatically rotated when they reach 10MB. The system keeps 5 backup files:
- `logs/aurora_qa.log` (current)
- `logs/aurora_qa.log.1` (previous)
- `logs/aurora_qa.log.2` (older)
- etc.

### Debug Mode

For maximum verbosity, set:

```env
LOG_LEVEL=DEBUG
```

This will show:
- All data transformations
- Sample values and statistics
- Full API request/response details
- Detailed timing information
- Complete error traces

### Disable File Logging

To disable file logging and only use console output, set in `.env`:

```env
LOG_FILE=
```

---

## External API Integration

### 1. Aurora Messages API

**Purpose**: Source of message data for the QA system.

**Endpoint**: `https://november7-730026606190.europe-west1.run.app/messages`

**Method**: GET

**Response Format:**
```json
{
  "total": 150,
  "items": [
    {
      "id": "msg_123",
      "user_id": "user_456",
      "message": "I'm going to Paris",
      "user_name": "Sophia",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    ...
  ]
}
```

**Usage**: Called once during startup to fetch all messages.

**Error Handling**: 
- HTTP errors are logged and re-raised
- Handles 307 redirects automatically with `follow_redirects=True`

### 2. HuggingFace Inference API

**Purpose**: Generate text embeddings for semantic search.

**Base URL**: `https://router.huggingface.co/hf-inference`

**Endpoint**: `/models/{model_name}`

**Method**: POST

**Authentication**: Bearer token in `Authorization` header

**Request Format:**
```json
{
  "inputs": "text to embed"  // Single text
}
// or
{
  "inputs": ["text1", "text2", ...]  // Batch
}
```

**Response Format:**
```json
[0.123, -0.456, 0.789, ...]  // Single text (384 dimensions)
// or
[[0.123, ...], [0.456, ...], ...]  // Batch
```

**Model**: `BAAI/bge-small-en-v1.5`
- **Dimensions**: 384
- **Language**: English
- **Size**: Small, efficient
- **Use Case**: Semantic similarity search

**Usage**:
- **Startup**: Batch embedding of all messages
- **Request**: Single embedding of user question

**Error Handling**:
- **503 (Model Loading)**: Wait 10 seconds and retry once
- **Other HTTP errors**: Log and raise exception
- **Timeout**: 60s for single, 120s for batch

### 3. Groq API

**Purpose**: Extract factual answers from questions and retrieved messages.

**Base URL**: `https://api.groq.com/openai/v1`

**Endpoint**: `/chat/completions`

**Method**: POST

**Authentication**: Bearer token in `Authorization` header

**Request Format:**
```json
{
  "model": "llama3-8b-instruct",
  "messages": [
    {
      "role": "system",
      "content": "You are an information extraction system..."
    },
    {
      "role": "user",
      "content": "Question: ...\n\nRelevant messages:\n..."
    }
  ],
  "temperature": 0.0,
  "max_tokens": 150
}
```

**Response Format:**
```json
{
  "choices": [
    {
      "message": {
        "content": "Sophia is going to Paris."
      }
    }
  ]
}
```

**Model**: `llama3-8b-instruct`
- **Provider**: Groq (fast inference)
- **Type**: Instruction-tuned LLM
- **API**: OpenAI-compatible

**Usage**: Called for each question to extract answers.

**Error Handling**:
- **HTTP errors**: Log and return "No information found."
- **All exceptions**: Catch and return safe fallback
- **Timeout**: 60 seconds

---

## Performance & Optimization

### Index Building Performance

**Factors:**
- **Number of messages**: Linear scaling with dataset size
- **HuggingFace API latency**: Primary bottleneck (network I/O)
- **Embedding dimension**: Fixed at 384 (model-dependent)

**Optimization:**
- **Batch embedding**: All messages embedded in single API call
- **In-memory index**: FAISS index stored in memory for fast access

**Typical Performance:**
- **100 messages**: ~5-10 seconds
- **500 messages**: ~15-30 seconds
- **1000 messages**: ~30-60 seconds

### Query Performance

**Factors:**
- **Question embedding**: HuggingFace API latency (~0.5-2s)
- **FAISS search**: In-memory, very fast (<0.01s)
- **LLM extraction**: Groq API latency (~1-3s)

**Optimization:**
- **FAISS IndexFlatL2**: Exact search (no approximation overhead)
- **Top-K limiting**: Only retrieve necessary messages
- **Async operations**: Non-blocking I/O for API calls

**Typical Performance:**
- **Total query time**: ~2-5 seconds
- **Embedding**: ~0.5-2s
- **Search**: <0.01s
- **LLM**: ~1-3s

### Memory Usage

**Components:**
- **Message objects**: ~1-10 KB per message (depends on text length)
- **Embeddings**: 384 floats × 4 bytes = 1.5 KB per message
- **FAISS index**: ~1.5 KB per message (embeddings + overhead)

**Total Memory (approximate):**
- **100 messages**: ~300 KB
- **500 messages**: ~1.5 MB
- **1000 messages**: ~3 MB

**Note**: Memory usage is minimal for typical dataset sizes.

### Scalability Considerations

**Current Architecture:**
- **Suitable for**: Small to medium datasets (<10K messages)
- **Bottleneck**: HuggingFace API rate limits and latency
- **Index type**: `IndexFlatL2` (exact search, O(n) complexity)

**Scaling Options:**
1. **Approximate search**: Use `IndexIVFFlat` or `IndexHNSW` for faster search on large datasets
2. **Caching**: Cache question embeddings for repeated queries
3. **Batch processing**: Process multiple questions in parallel
4. **Distributed index**: Split index across multiple instances (advanced)

---

## Deployment

### Docker Deployment

**Build the image:**
```bash
docker build -t aurora-qa .
```

**Run the container:**
```bash
docker run -p 8000:8000 \
  -e HF_API_TOKEN=your_hf_token_here \
  -e GROQ_API_KEY=your_groq_key_here \
  -e MESSAGES_API_URL=https://november7-730026606190.europe-west1.run.app/messages \
  aurora-qa
```

**Using Docker Compose:**
```yaml
version: '3.8'
services:
  aurora-qa:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HF_API_TOKEN=${HF_API_TOKEN}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - MESSAGES_API_URL=${MESSAGES_API_URL}
    env_file:
      - .env
```

### Deploy to Railway

1. Connect your repository to Railway
2. Set environment variables in Railway dashboard:
   - `HF_API_TOKEN` (required)
   - `GROQ_API_KEY` (required)
   - `MESSAGES_API_URL` (optional, has default)
3. Railway will automatically detect the Dockerfile and deploy

The `PORT` environment variable will be automatically set by Railway.

### Deploy to Heroku

1. Create `Procfile`:
   ```
   web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
   ```

2. Set environment variables in Heroku dashboard or CLI:
   ```bash
   heroku config:set HF_API_TOKEN=your_token
   heroku config:set GROQ_API_KEY=your_key
   ```

3. Deploy:
   ```bash
   git push heroku main
   ```

### Production Considerations

1. **Environment Variables**: Never commit `.env` file to version control
2. **Logging**: Configure appropriate log levels and file rotation
3. **Health Checks**: Use `/health` endpoint for load balancer health checks
4. **Monitoring**: Set up monitoring for API response times and error rates
5. **Rate Limiting**: Consider adding rate limiting for production use
6. **CORS**: Update CORS settings to allow only specific origins

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Index Not Ready Error

**Error**: `RuntimeError: Index not ready. Ensure index is built on startup.`

**Causes:**
- Startup failed silently
- Index building was interrupted
- Application restarted but index not rebuilt

**Solutions:**
1. Check application logs for startup errors
2. Verify all environment variables are set correctly
3. Ensure Aurora API is accessible
4. Restart the application

#### 2. HuggingFace API 503 Error

**Error**: `HTTP 503: Model is loading`

**Causes:**
- HuggingFace model is cold-starting
- Model endpoint is temporarily unavailable

**Solutions:**
1. Wait 10-30 seconds (automatic retry handles this)
2. Check HuggingFace status page
3. Verify API token is valid
4. Try a different model if persistent

#### 3. Groq API Errors

**Error**: `HTTP error extracting answer` or `No information found.`

**Causes:**
- Invalid API key
- Rate limiting
- Model unavailable
- Network issues

**Solutions:**
1. Verify `GROQ_API_KEY` is correct
2. Check Groq API status
3. Verify model name (`llama3-8b-instruct`)
4. Check network connectivity

#### 4. Empty Question Error

**Error**: `HTTP 400: Question cannot be empty`

**Causes:**
- Client sent empty or whitespace-only question
- Request validation failed

**Solutions:**
1. Ensure question field is non-empty
2. Strip whitespace before sending
3. Validate input on client side

#### 5. No Messages Retrieved

**Symptom**: Always returns "No information found."

**Causes:**
- Index is empty (startup failed)
- Question embedding doesn't match any messages
- Top-K is too small

**Solutions:**
1. Check startup logs for index building success
2. Verify messages were fetched during startup
3. Try increasing `TOP_K` value
4. Test with different question phrasings

#### 6. Slow Response Times

**Symptom**: Requests take >10 seconds

**Causes:**
- HuggingFace API latency
- Groq API latency
- Network issues
- High load on external APIs

**Solutions:**
1. Check external API status pages
2. Monitor network latency
3. Consider caching for repeated questions
4. Use faster embedding models (if available)

#### 7. Import Errors

**Error**: `ImportError: No module named 'app'`

**Causes:**
- Incorrect Python path
- Virtual environment not activated
- Missing dependencies

**Solutions:**
1. Ensure virtual environment is activated
2. Install dependencies: `pip install -r requirements.txt`
3. Run from project root directory
4. Check Python version (3.11+)

#### 8. FAISS Library Errors

**Error**: `ImportError: dlopen(...) Library not loaded`

**Causes:**
- NumPy/FAISS library linking issues (macOS)
- Incompatible library versions
- Conda environment conflicts

**Solutions:**
1. Use fresh Python virtual environment (not conda)
2. Reinstall: `pip install --upgrade numpy faiss-cpu`
3. On macOS, may need to set `DYLD_LIBRARY_PATH`
4. Use `start.sh` script which handles library paths

### Debugging Tips

1. **Enable verbose logging**: Set `LOG_LEVEL=DEBUG` in `.env`
2. **Check health endpoint**: `GET /health` to verify index status
3. **Test individual components**: Embed a test question, search index manually
4. **Monitor API responses**: Log external API requests/responses
5. **Use API docs**: FastAPI auto-generated docs at `/docs` for testing

### Health Check Endpoint

**Endpoint**: `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "index_ready": true
}
```

**Use Cases:**
- Verify application is running
- Check if index is ready
- Monitor application health
- Load balancer health checks

---

## Development Guide

### Development Setup

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd Aurora_Assessment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run in development mode**
   ```bash
   uvicorn app.main:app --reload
   ```

### Code Structure and Conventions

**File Organization:**
- `app/api/`: API endpoints and routes
- `app/core/`: Core functionality (embeddings, LLM, config)
- `app/models/`: Pydantic data models
- `app/services/`: Business logic and orchestration

**Code Style:**
- Type hints throughout
- Async/await for I/O operations
- Comprehensive error handling
- Detailed logging
- Docstrings for all functions

**Naming Conventions:**
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Testing Approach

**Running Tests:**
```bash
pytest tests/
```

**Test Structure:**
- Unit tests for individual components
- Integration tests for API endpoints
- Mock external API calls

### Contributing Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes**: Follow code style and conventions
4. **Add tests**: Ensure new features are tested
5. **Update documentation**: Update README if needed
6. **Commit changes**: Use clear commit messages
7. **Push to branch**: `git push origin feature-name`
8. **Create pull request**: Describe changes clearly

### Code Quality

The codebase follows:
- **Type hints**: Full type annotations throughout
- **Async/await**: All I/O operations are asynchronous
- **Error handling**: Proper error handling and logging
- **Separation of concerns**: Clean separation of responsibilities
- **Production-ready patterns**: Defensive programming, fallbacks

---

## License

This project is part of Aurora's take-home assignment.

## Author

Built as a demonstration of modern AI system design, retrieval optimization, clean API construction, and production-quality architecture using free, open-source tools.

---

## Additional Resources

- **Backend Flow Documentation**: See `BACKEND_FLOW.md` for detailed workflow documentation
- **Logging Guide**: See `LOGGING.md` for comprehensive logging usage guide
- **API Documentation**: Interactive docs available at `http://localhost:8000/docs` when running
- **Source Code**: Component docstrings and inline comments provide additional details

---

## Free Tier Architecture

This implementation uses **100% free APIs**:
- **HuggingFace Inference API**: Free embeddings with generous rate limits
- **Groq API**: Free LLM inference with fast response times
- **FAISS**: Local CPU-based vector search (no API costs)

No OpenAI costs, no local GPU required, fully cloud-based and scalable.
