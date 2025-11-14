# Backend Flow Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Startup Sequence](#startup-sequence)
4. [Request Processing Flow](#request-processing-flow)
5. [Component Details](#component-details)
6. [Data Flow Diagrams](#data-flow-diagrams)
7. [Error Handling](#error-handling)
8. [Configuration](#configuration)
9. [External APIs](#external-apis)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

The Aurora QA System is a production-ready question-answering API that implements a **Retrieval-Augmented Generation (RAG)** pipeline. The system answers natural language questions about users by:

1. Retrieving relevant messages from Aurora's public API
2. Building a vector index using FAISS for efficient similarity search
3. Using semantic search to find the most relevant messages for a question
4. Extracting factual answers using a Large Language Model (LLM)

The backend is built with **FastAPI** and uses a **100% free architecture** with:
- **HuggingFace Inference API** for text embeddings
- **Groq API** for LLM-based answer extraction
- **FAISS** for local vector similarity search

---

## Architecture Overview

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

### Technology Stack

- **Framework**: FastAPI (Python 3.11+)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace Inference API (`BAAI/bge-small-en-v1.5`)
- **LLM**: Groq API (`llama3-8b-instruct`)
- **HTTP Client**: httpx (async)
- **Validation**: Pydantic
- **Configuration**: pydantic-settings

### Component Layers

1. **API Layer**: FastAPI routes, request/response validation
2. **Service Layer**: Business logic orchestration (QA engine, retrieval, indexing)
3. **Core Layer**: External API integrations (embeddings, LLM)
4. **Model Layer**: Pydantic models for data validation
5. **Configuration Layer**: Environment-based settings management

---

## Startup Sequence

The application follows a specific startup sequence to build the FAISS index before accepting requests.

### Startup Flow Diagram

```
Application Start
       │
       ▼
┌──────────────────────┐
│  FastAPI App Init    │
│  • Create app        │
│  • Configure CORS     │
│  • Register routes    │
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
│  • Create VectorIndex│
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

### Detailed Startup Steps

#### Step 1: Application Initialization (`app/main.py`)

```python
# FastAPI app creation
app = FastAPI(
    title="Aurora QA System",
    description="Question-answering API for Aurora's take-home assignment",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(ask_router, prefix="/api", tags=["QA"])
```

#### Step 2: Startup Event Handler

The `@app.on_event("startup")` decorator triggers the index building process:

```python
@app.on_event("startup")
async def startup_event():
    """Build FAISS index on application startup."""
    logger.info("Starting up Aurora QA System...")
    try:
        index = get_index()
        await index.build_index()
        logger.info("Startup complete. Index is ready.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise  # Fail fast if index cannot be built
```

**Key Points:**
- Startup is **blocking**: The application will not accept requests until the index is built
- **Fail-fast**: If index building fails, the application raises an exception
- **Single instance**: Uses a global `VectorIndex` instance via `get_index()`

#### Step 3: Message Ingestion (`app/services/ingestion.py`)

```python
async def fetch_messages() -> List[Message]:
    """
    Fetch messages from Aurora's public API.
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(settings.messages_api_url)
        response.raise_for_status()
        
        data = response.json()
        
        # API returns {total: int, items: [...]}
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        else:
            items = data if isinstance(data, list) else []
        
        # Parse into Message objects
        messages = [Message(**msg) for msg in items]
        return messages
```

**Process:**
1. HTTP GET request to Aurora API endpoint
2. Parse JSON response (handles both `{items: [...]}` and `[...]` formats)
3. Convert each item to a `Message` Pydantic model
4. Return list of `Message` objects

**Error Handling:**
- HTTP errors are logged and re-raised
- JSON parsing errors are caught and re-raised
- `follow_redirects=True` handles 307 redirects

#### Step 4: Message Embedding (`app/core/embeddings.py`)

```python
async def embed_messages(messages: List[Message]) -> np.ndarray:
    """
    Embed all messages in batch using HuggingFace Inference API.
    """
    texts = [msg.text for msg in messages]
    
    url = f"{HF_API_BASE}/models/{settings.hf_embedding_model}"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": texts}  # Batch request
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        embeddings_data = response.json()
        embeddings = np.array(embeddings_data, dtype=np.float32)
        
        return embeddings  # Shape: (num_messages, embedding_dim)
```

**Process:**
1. Extract text content from all messages
2. Send batch POST request to HuggingFace Inference API
3. Convert response to numpy array of shape `(N, embedding_dim)`
4. Return embeddings array

**Retry Logic:**
- If API returns 503 (model loading), wait 10 seconds and retry once
- Timeout set to 120 seconds for large batches

#### Step 5: FAISS Index Construction (`app/services/indexing.py`)

```python
async def build_index(self) -> None:
    """
    Build FAISS index from message embeddings.
    """
    # 1. Fetch messages
    messages = await fetch_messages()
    self.messages = messages
    
    # 2. Embed all messages
    embeddings = await embed_messages(messages)
    self.embeddings = embeddings
    
    # 3. Get embedding dimension
    dimension = embeddings.shape[1]
    
    # 4. Create FAISS index (L2 distance)
    self.index = faiss.IndexFlatL2(dimension)
    
    # 5. Add embeddings to index
    self.index.add(embeddings)
    
    # 6. Mark as ready
    self._is_built = True
```

**Index Type: `IndexFlatL2`**
- **L2 Distance**: Euclidean distance between vectors
- **Exact Search**: No approximation, guarantees exact nearest neighbors
- **Memory**: Stores all vectors in memory
- **Speed**: O(n) search time, suitable for moderate dataset sizes

**Index State:**
- `_is_built`: Boolean flag indicating index readiness
- `is_ready()`: Public method to check if index is ready for queries

### Startup Timing

Typical startup time breakdown:
- **Message Fetching**: 1-3 seconds (depends on API response time)
- **Embedding Generation**: 5-15 seconds (depends on number of messages and HuggingFace API latency)
- **Index Construction**: < 1 second (in-memory operation)
- **Total**: ~10-20 seconds for typical datasets

---

## Request Processing Flow

When a client sends a question to `/api/ask`, the system processes it through a multi-stage pipeline.

### Request Flow Diagram

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
│  QA Engine (app/services/qa_engine.py)   │
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

### Detailed Request Processing Steps

#### Step 1: API Request Validation (`app/api/ask.py`)

```python
@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    # Pydantic automatically validates request schema
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Call QA engine
    answer = await answer_question(question)
    
    # Return response
    return AskResponse(answer=answer)
```

**Validation:**
- **Pydantic**: Automatically validates `AskRequest` model (ensures `question` is a non-empty string)
- **Manual Check**: Additional check for empty/whitespace-only questions
- **Error Response**: Returns 400 Bad Request for invalid input

#### Step 2: QA Engine Orchestration (`app/services/qa_engine.py`)

```python
async def answer_question(question: str) -> str:
    """
    High-level QA engine orchestrator.
    """
    # Step 1: Retrieve top-K relevant messages
    messages = await retrieve_relevant_messages(question)
    
    if not messages:
        return "No information found."
    
    # Step 2: Extract answer using LLM
    answer = await extract_answer(question, messages)
    
    return answer
```

**Orchestration Logic:**
- **Sequential Processing**: Retrieval happens first, then LLM extraction
- **Early Exit**: If no messages retrieved, return immediately
- **Error Handling**: Catches exceptions and returns safe fallback

#### Step 3: Question Embedding (`app/core/embeddings.py`)

```python
async def embed_text(text: str) -> np.ndarray:
    """
    Embed a single question string.
    """
    url = f"{HF_API_BASE}/models/{settings.hf_embedding_model}"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        embedding = response.json()
        # Handle response format
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                embedding = embedding[0]
        
        return np.array(embedding, dtype=np.float32)
```

**Process:**
1. Single text embedding request to HuggingFace API
2. Handle both single-item and nested list responses
3. Convert to numpy array of shape `(embedding_dim,)`

#### Step 4: FAISS Vector Search (`app/services/indexing.py`)

```python
def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Message, float]]:
    """
    Search FAISS index for top-K most similar messages.
    """
    # Reshape query to (1, embedding_dim) for FAISS
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search
    k = min(k, len(self.messages))
    distances, indices = self.index.search(query_embedding, k)
    
    # Return messages with their distances
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(self.messages):
            results.append((self.messages[idx], float(distance)))
    
    return results
```

**Search Process:**
1. Reshape query embedding to `(1, embedding_dim)` for FAISS API
2. Call `index.search()` which returns:
   - `distances`: L2 distances (lower = more similar)
   - `indices`: Indices of matching messages
3. Map indices back to `Message` objects
4. Return list of `(Message, distance)` tuples

**Distance Metric:**
- **L2 (Euclidean) Distance**: `sqrt(sum((a_i - b_i)^2))`
- Lower distance = higher similarity
- Results are automatically sorted by distance (ascending)

#### Step 5: LLM Answer Extraction (`app/core/llm.py`)

```python
async def extract_answer(question: str, messages: List[Message]) -> str:
    """
    Extract factual answer using Groq API.
    """
    # Build prompt
    user_prompt = build_user_prompt(question, messages)
    
    # Prepare API request
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,  # Deterministic
        "max_tokens": 150    # Short answers
    }
    
    # Call Groq API
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{GROQ_API_BASE}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"}
        )
        response.raise_for_status()
        
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        
        # Ensure sentence ending
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer
```

**Prompt Structure:**
```
System: You are an information extraction system. 
        You must answer strictly based on the provided messages. 
        If the information is not present, reply: "No information found."

User: Question: {question}

        Relevant messages:

        Message 1 (User: user_123): {message_text}
        Message 2 (User: user_456): {message_text}
        ...

        Answer in one short factually correct sentence.
```

**LLM Configuration:**
- **Temperature**: 0.0 (deterministic responses)
- **Max Tokens**: 150 (forces concise answers)
- **Model**: `llama3-8b-instruct` (via Groq API)

**Answer Formatting:**
- Strips whitespace
- Ensures sentence ends with punctuation
- Returns single sentence

### Request Processing Timing

Typical request processing time:
- **Question Embedding**: 0.5-2 seconds (HuggingFace API latency)
- **FAISS Search**: < 0.01 seconds (in-memory operation)
- **LLM Extraction**: 1-3 seconds (Groq API latency)
- **Total**: ~2-5 seconds per request

---

## Component Details

### API Layer

#### `app/main.py`

**Purpose**: FastAPI application entry point and configuration.

**Key Responsibilities:**
- Create FastAPI app instance
- Configure CORS middleware
- Register API routes
- Define startup event handler
- Provide health check endpoint

**Key Code:**
```python
app = FastAPI(title="Aurora QA System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
app.include_router(ask_router, prefix="/api", tags=["QA"])

@app.on_event("startup")
async def startup_event():
    index = get_index()
    await index.build_index()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "index_ready": get_index().is_ready()}
```

**Dependencies:**
- `app.api.ask`: API router
- `app.services.indexing`: Index management

#### `app/api/ask.py`

**Purpose**: Handle `/api/ask` POST endpoint.

**Input**: `AskRequest` (Pydantic model)
```python
class AskRequest(BaseModel):
    question: str  # min_length=1
```

**Output**: `AskResponse` (Pydantic model)
```python
class AskResponse(BaseModel):
    answer: str
```

**Error Handling:**
- **400 Bad Request**: Empty or invalid question
- **500 Internal Server Error**: Processing failures

**Key Code:**
```python
@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    answer = await answer_question(question)
    return AskResponse(answer=answer)
```

**Dependencies:**
- `app.services.qa_engine`: Core QA logic
- `app.models.request`: Request validation
- `app.models.response`: Response serialization

### Service Layer

#### `app/services/qa_engine.py`

**Purpose**: High-level orchestrator for question answering.

**Function**: `answer_question(question: str) -> str`

**Process:**
1. Call `retrieve_relevant_messages()` to get top-K messages
2. If no messages, return "No information found."
3. Call `extract_answer()` with question and messages
4. Return extracted answer

**Error Handling:**
- Catches all exceptions
- Returns "No information found." as safe fallback
- Logs errors for debugging

**Dependencies:**
- `app.services.retrieval`: Message retrieval
- `app.core.llm`: Answer extraction

#### `app/services/retrieval.py`

**Purpose**: Retrieve relevant messages using vector similarity search.

**Function**: `retrieve_relevant_messages(question: str, k: int = None) -> List[Message]`

**Process:**
1. Embed question using `embed_text()`
2. Get FAISS index instance
3. Verify index is ready
4. Search index for top-K messages
5. Return list of `Message` objects

**Key Code:**
```python
async def retrieve_relevant_messages(question: str, k: int = None) -> List[Message]:
    if k is None:
        k = settings.top_k
    
    question_embedding = await embed_text(question)
    index = get_index()
    
    if not index.is_ready():
        raise RuntimeError("Index not ready.")
    
    results = index.search(question_embedding, k=k)
    messages = [msg for msg, _ in results]
    
    return messages
```

**Dependencies:**
- `app.core.embeddings`: Question embedding
- `app.services.indexing`: FAISS index access
- `app.core.config`: Configuration (top_k)

#### `app/services/indexing.py`

**Purpose**: Manage FAISS vector index for message embeddings.

**Class**: `VectorIndex`

**Key Methods:**
- `build_index()`: Build index from messages (startup)
- `search(query_embedding, k)`: Search for top-K similar messages
- `is_ready()`: Check if index is built

**Index Type**: `faiss.IndexFlatL2`
- Exact L2 distance search
- In-memory storage
- O(n) search complexity

**State Management:**
- `_is_built`: Boolean flag
- `messages`: List of original messages
- `embeddings`: Numpy array of embeddings
- `index`: FAISS index instance

**Key Code:**
```python
class VectorIndex:
    def __init__(self):
        self.messages: List[Message] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self._is_built = False
    
    async def build_index(self) -> None:
        messages = await fetch_messages()
        embeddings = await embed_messages(messages)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self._is_built = True
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [(self.messages[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]
```

**Dependencies:**
- `app.services.ingestion`: Message fetching
- `app.core.embeddings`: Message embedding
- `faiss`: Vector search library

#### `app/services/ingestion.py`

**Purpose**: Fetch messages from Aurora's public API.

**Function**: `fetch_messages() -> List[Message]`

**Process:**
1. HTTP GET request to Aurora API
2. Parse JSON response (handles `{items: [...]}` format)
3. Convert each item to `Message` Pydantic model
4. Return list of messages

**Error Handling:**
- HTTP errors: Logged and re-raised
- JSON parsing errors: Caught and re-raised
- Handles 307 redirects with `follow_redirects=True`

**Key Code:**
```python
async def fetch_messages() -> List[Message]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(settings.messages_api_url)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        else:
            items = data if isinstance(data, list) else []
        
        messages = [Message(**msg) for msg in items]
        return messages
```

**Dependencies:**
- `app.models.message`: Message model
- `app.core.config`: API URL configuration
- `httpx`: HTTP client

### Core Layer

#### `app/core/embeddings.py`

**Purpose**: Generate text embeddings using HuggingFace Inference API.

**Functions:**
- `embed_text(text: str) -> np.ndarray`: Embed single text
- `embed_messages(messages: List[Message]) -> np.ndarray`: Embed batch of messages

**API Endpoint**: `https://router.huggingface.co/hf-inference/models/{model_name}`

**Model**: `BAAI/bge-small-en-v1.5` (default)
- Small, efficient embedding model
- 384-dimensional embeddings
- Optimized for English text

**Request Format:**
```json
{
  "inputs": "text to embed"  // or ["text1", "text2", ...] for batch
}
```

**Response Format:**
```json
[0.123, -0.456, 0.789, ...]  // Single text
// or
[[0.123, ...], [0.456, ...], ...]  // Batch
```

**Retry Logic:**
- If API returns 503 (model loading), wait 10 seconds and retry once
- Timeout: 60s for single, 120s for batch

**Key Code:**
```python
async def embed_text(text: str) -> np.ndarray:
    url = f"{HF_API_BASE}/models/{settings.hf_embedding_model}"
    headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
    payload = {"inputs": text}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        embedding = response.json()
        return np.array(embedding, dtype=np.float32)
```

**Dependencies:**
- `app.core.config`: API token and model name
- `httpx`: HTTP client
- `numpy`: Array operations

#### `app/core/llm.py`

**Purpose**: Extract factual answers using Groq API (LLM).

**Functions:**
- `build_user_prompt(question, messages) -> str`: Build prompt template
- `extract_answer(question, messages) -> str`: Extract answer from LLM

**API Endpoint**: `https://api.groq.com/openai/v1/chat/completions`

**Model**: `llama3-8b-instruct` (default)
- Fast inference via Groq
- Instruction-tuned for following prompts
- OpenAI-compatible API

**System Prompt:**
```
You are an information extraction system. 
You must answer strictly based on the provided messages. 
If the information is not present, reply: "No information found."
```

**User Prompt Template:**
```
Question: {question}

Relevant messages:

Message 1 (User: {user_id}): {message_text}
Message 2 (User: {user_id}): {message_text}
...

Answer in one short factually correct sentence.
```

**Request Format:**
```json
{
  "model": "llama3-8b-instruct",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.0,
  "max_tokens": 150
}
```

**Response Format:**
```json
{
  "choices": [{
    "message": {
      "content": "Answer text here."
    }
  }]
}
```

**Error Handling:**
- HTTP errors: Logged, returns "No information found."
- All exceptions: Caught, returns safe fallback

**Key Code:**
```python
async def extract_answer(question: str, messages: List[Message]) -> str:
    user_prompt = build_user_prompt(question, messages)
    
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{GROQ_API_BASE}/chat/completions", ...)
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer
```

**Dependencies:**
- `app.core.config`: API key and model name
- `app.models.message`: Message objects
- `httpx`: HTTP client

#### `app/core/config.py`

**Purpose**: Manage application configuration from environment variables.

**Class**: `Settings` (Pydantic BaseSettings)

**Configuration Variables:**

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `hf_api_token` | str | Yes | - | HuggingFace API token |
| `groq_api_key` | str | Yes | - | Groq API key |
| `messages_api_url` | str | No | `https://november7-730026606190.europe-west1.run.app/messages` | Aurora API endpoint |
| `hf_embedding_model` | str | No | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `groq_model` | str | No | `llama3-8b-instruct` | Groq LLM model |
| `top_k` | int | No | `5` | Number of messages to retrieve |
| `port` | int | No | `8000` | Server port |

**Configuration Loading:**
- Loads from `.env` file
- Environment variables override file values
- Case-insensitive variable names
- Validates types using Pydantic

**Key Code:**
```python
class Settings(BaseSettings):
    hf_api_token: str
    groq_api_key: str
    messages_api_url: str = "https://november7-730026606190.europe-west1.run.app/messages"
    hf_embedding_model: str = "BAAI/bge-small-en-v1.5"
    groq_model: str = "llama3-8b-instruct"
    top_k: int = 5
    port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()
```

**Dependencies:**
- `pydantic_settings`: Settings management

### Model Layer

#### `app/models/message.py`

**Purpose**: Represent message data from Aurora API.

**Model**: `Message`

**Fields:**
- `id: str`: Message ID
- `user_id: str`: User identifier
- `message: str`: Message text content
- `user_name: Optional[str]`: User name (optional)
- `timestamp: Optional[datetime]`: Message timestamp (optional)

**Property:**
- `text: str`: Alias for `message` field (backward compatibility)

**Key Code:**
```python
class Message(BaseModel):
    id: str
    user_id: str
    message: str
    user_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    @property
    def text(self) -> str:
        return self.message
```

#### `app/models/request.py`

**Purpose**: Validate incoming API requests.

**Model**: `AskRequest`

**Fields:**
- `question: str`: User's question (min_length=1)

**Key Code:**
```python
class AskRequest(BaseModel):
    question: str = Field(..., description="The question to answer", min_length=1)
```

#### `app/models/response.py`

**Purpose**: Structure API responses.

**Model**: `AskResponse`

**Fields:**
- `answer: str`: Extracted answer

**Key Code:**
```python
class AskResponse(BaseModel):
    answer: str
```

---

## Data Flow Diagrams

### Startup Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Startup Data Flow                         │
└─────────────────────────────────────────────────────────────────┘

1. Application Start
   │
   ▼
2. FastAPI App Initialization
   │
   ▼
3. Startup Event Triggered
   │
   ▼
4. fetch_messages()
   │
   │ HTTP GET → Aurora API
   │
   │ JSON Response: { "total": N, "items": [...] }
   │
   ▼
5. Parse JSON → List[Message]
   │
   │ Message objects:
   │   - id: "msg_123"
   │   - user_id: "user_456"
   │   - message: "I'm going to Paris"
   │   - user_name: "Sophia"
   │   - timestamp: datetime(...)
   │
   ▼
6. embed_messages(messages)
   │
   │ Extract texts: ["I'm going to Paris", "Dinner at 8pm", ...]
   │
   │ HTTP POST → HuggingFace API
   │   { "inputs": ["text1", "text2", ...] }
   │
   │ JSON Response: [[0.123, -0.456, ...], [0.789, ...], ...]
   │
   ▼
7. Convert to numpy array
   │
   │ Shape: (N, 384)  # N messages, 384-dimensional embeddings
   │
   ▼
8. Build FAISS Index
   │
   │ Create IndexFlatL2(dimension=384)
   │ Add embeddings to index
   │
   ▼
9. Index Ready
   │
   │ _is_built = True
   │ index.ntotal = N
   │
   ▼
10. Application Ready to Accept Requests
```

### Request Processing Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Request Processing Data Flow                  │
└─────────────────────────────────────────────────────────────────┘

1. Client Request
   │
   │ POST /api/ask
   │ { "question": "Where is Sophia going?" }
   │
   ▼
2. Request Validation
   │
   │ AskRequest(question="Where is Sophia going?")
   │
   ▼
3. QA Engine: answer_question()
   │
   ▼
4. Retrieval: retrieve_relevant_messages()
   │
   │ 4a. Embed question
   │     │
   │     │ embed_text("Where is Sophia going?")
   │     │
   │     │ HTTP POST → HuggingFace API
   │     │   { "inputs": "Where is Sophia going?" }
   │     │
   │     │ JSON Response: [0.123, -0.456, 0.789, ...]
   │     │
   │     ▼
   │     numpy array: shape (384,)
   │
   │ 4b. Search FAISS index
   │     │
   │     │ index.search(query_embedding, k=5)
   │     │
   │     │ Returns:
   │     │   distances: [0.12, 0.34, 0.56, 0.78, 0.90]
   │     │   indices: [42, 15, 88, 123, 7]
   │     │
   │     ▼
   │     List[Tuple[Message, float]]:
   │       - (Message(id="msg_42", message="I'm going to Paris", ...), 0.12)
   │       - (Message(id="msg_15", message="Paris trip confirmed", ...), 0.34)
   │       - ...
   │
   ▼
5. Extract messages (ignore distances)
   │
   │ List[Message]: [Message(...), Message(...), ...]
   │
   ▼
6. LLM: extract_answer()
   │
   │ 6a. Build prompt
   │     │
   │     │ System: "You are an information extraction system..."
   │     │ User: "Question: Where is Sophia going?
   │     │
   │     │        Relevant messages:
   │     │        Message 1 (User: user_123): I'm going to Paris
   │     │        Message 2 (User: user_123): Paris trip confirmed
   │     │        ...
   │     │
   │     │        Answer in one short factually correct sentence."
   │
   │ 6b. Call Groq API
   │     │
   │     │ HTTP POST → Groq API
   │     │   {
   │     │     "model": "llama3-8b-instruct",
   │     │     "messages": [...],
   │     │     "temperature": 0.0,
   │     │     "max_tokens": 150
   │     │   }
   │     │
   │     │ JSON Response:
   │     │   {
   │     │     "choices": [{
   │     │       "message": {
   │     │         "content": "Sophia is going to Paris."
   │     │       }
   │     │     }]
   │     │   }
   │
   ▼
7. Format answer
   │
   │ "Sophia is going to Paris."
   │
   ▼
8. Response
   │
   │ { "answer": "Sophia is going to Paris." }
   │
   ▼
9. Client receives answer
```

### Embedding Pipeline

```
Text Input
    │
    ▼
┌─────────────────┐
│  Preprocessing   │
│  (if needed)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HuggingFace    │
│  Inference API  │
│                 │
│  Model:         │
│  bge-small-en   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  384-dim Vector │
│  (numpy array)  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Normalization  │
│  (if needed)    │
└────────┬────────┘
         │
         ▼
    FAISS Index
```

### Retrieval Pipeline

```
Question
    │
    ▼
┌─────────────────┐
│  Embed Question │
│  (HuggingFace)  │
└────────┬────────┘
         │
         │ query_vector: (384,)
         ▼
┌─────────────────┐
│  FAISS Search   │
│                 │
│  L2 Distance:   │
│  sqrt(Σ(a-b)²)  │
└────────┬────────┘
         │
         │ Top-K results:
         │ [(msg1, dist1), (msg2, dist2), ...]
         ▼
┌─────────────────┐
│  Sort by        │
│  Distance       │
│  (ascending)    │
└────────┬────────┘
         │
         │ List[Message]
         ▼
    Top-K Messages
```

### LLM Extraction Pipeline

```
Question + Messages
    │
    ▼
┌─────────────────┐
│  Build Prompt   │
│                 │
│  System:        │
│  "You are..."   │
│                 │
│  User:          │
│  "Question: ..."│
│  "Messages: ..."│
└────────┬────────┘
         │
         │ prompt: str
         ▼
┌─────────────────┐
│  Groq API Call  │
│                 │
│  Model:         │
│  llama3-8b-     │
│  instruct       │
│                 │
│  Temperature: 0 │
│  Max Tokens: 150│
└────────┬────────┘
         │
         │ response: JSON
         ▼
┌─────────────────┐
│  Extract Answer │
│  from response  │
└────────┬────────┘
         │
         │ answer: str
         ▼
┌─────────────────┐
│  Format Answer  │
│  (ensure        │
│  punctuation)   │
└────────┬────────┘
         │
         ▼
    Final Answer
```

---

## Error Handling

### Error Handling Strategy

The system implements a **defensive programming** approach with multiple layers of error handling:

1. **API Layer**: Validates input and returns appropriate HTTP status codes
2. **Service Layer**: Catches exceptions and returns safe fallbacks
3. **Core Layer**: Handles external API errors with retries and fallbacks

### Error Propagation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Error Propagation                        │
└─────────────────────────────────────────────────────────────┘

External API Error (HuggingFace/Groq)
    │
    ▼
httpx.HTTPStatusError / Exception
    │
    ▼
Core Layer (embeddings.py / llm.py)
    │
    │ Log error
    │
    ├─> Retry (if applicable)
    │   │
    │   └─> Success: Continue
    │   └─> Failure: Raise exception or return fallback
    │
    └─> Return fallback / Raise exception
        │
        ▼
Service Layer (qa_engine.py / retrieval.py)
    │
    │ Log error
    │
    ├─> Return "No information found." (qa_engine)
    │
    └─> Raise RuntimeError (retrieval)
        │
        ▼
API Layer (ask.py)
    │
    │ Log error
    │
    └─> Return HTTP 500 with error message
        │
        ▼
Client receives error response
```

### Error Types and Handling

#### 1. Input Validation Errors

**Location**: `app/api/ask.py`

**Errors:**
- Empty question
- Invalid request format

**Handling:**
```python
if not question:
    raise HTTPException(status_code=400, detail="Question cannot be empty")
```

**Response**: HTTP 400 Bad Request

#### 2. Index Not Ready

**Location**: `app/services/retrieval.py`

**Error**: Index not built during startup

**Handling:**
```python
if not index.is_ready():
    raise RuntimeError("Index not ready. Ensure index is built on startup.")
```

**Response**: HTTP 500 Internal Server Error

#### 3. Embedding API Errors

**Location**: `app/core/embeddings.py`

**Errors:**
- HTTP 503 (Model loading)
- HTTP 4xx/5xx (API errors)
- Network timeouts

**Handling:**
```python
# Retry logic for 503
if e.response.status_code == 503:
    await asyncio.sleep(10)
    # Retry once
    response = await client.post(...)
else:
    logger.error(f"HTTP error embedding text: {e}")
    raise
```

**Response**: Exception raised to caller

#### 4. LLM API Errors

**Location**: `app/core/llm.py`

**Errors:**
- HTTP 4xx/5xx (API errors)
- Network timeouts
- Invalid response format

**Handling:**
```python
except httpx.HTTPStatusError as e:
    logger.error(f"HTTP error extracting answer: {e}")
    return "No information found."  # Safe fallback
except Exception as e:
    logger.error(f"Error extracting answer: {e}")
    return "No information found."  # Safe fallback
```

**Response**: Returns "No information found." (does not raise exception)

#### 5. QA Engine Errors

**Location**: `app/services/qa_engine.py`

**Errors:**
- Retrieval failures
- LLM extraction failures
- Unexpected exceptions

**Handling:**
```python
try:
    messages = await retrieve_relevant_messages(question)
    if not messages:
        return "No information found."
    answer = await extract_answer(question, messages)
    return answer
except Exception as e:
    logger.error(f"Error in QA engine: {e}")
    return "No information found."  # Safe fallback
```

**Response**: Returns "No information found." (does not raise exception)

#### 6. Startup Errors

**Location**: `app/main.py`

**Errors:**
- Message fetching failures
- Embedding generation failures
- Index construction failures

**Handling:**
```python
try:
    index = get_index()
    await index.build_index()
    logger.info("Startup complete. Index is ready.")
except Exception as e:
    logger.error(f"Error during startup: {e}")
    raise  # Fail fast - application will not start
```

**Response**: Application fails to start (exception raised)

### Fallback Mechanisms

1. **LLM Extraction**: Returns "No information found." on any error
2. **QA Engine**: Returns "No information found." if retrieval fails or no messages found
3. **API Layer**: Returns HTTP 500 with error message for unexpected errors

### Logging Strategy

**Log Levels:**
- **INFO**: Normal operations (startup, request processing, successful operations)
- **WARNING**: Recoverable issues (model loading, retries)
- **ERROR**: Failures that require attention (API errors, exceptions)

**Log Format:**
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

**Example Logs:**
```
2024-01-15 10:30:45 - app.main - INFO - Starting up Aurora QA System...
2024-01-15 10:30:46 - app.services.ingestion - INFO - Fetched 150 messages from API
2024-01-15 10:30:50 - app.core.embeddings - INFO - Embedded 150 messages into (150, 384) array
2024-01-15 10:30:51 - app.services.indexing - INFO - Successfully built FAISS index with 150 vectors
2024-01-15 10:30:51 - app.main - INFO - Startup complete. Index is ready.
2024-01-15 10:31:00 - app.services.qa_engine - INFO - Processing question: Where is Sophia going?...
2024-01-15 10:31:01 - app.services.retrieval - INFO - Retrieved 5 relevant messages
2024-01-15 10:31:03 - app.core.llm - INFO - Extracted answer: Sophia is going to Paris.
```

---

## Configuration

### Environment Variables

All configuration is managed through environment variables loaded from a `.env` file.

#### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_API_TOKEN` | HuggingFace API token for embeddings | `hf_xxxxxxxxxxxxx` |
| `GROQ_API_KEY` | Groq API key for LLM | `gsk_xxxxxxxxxxxxx` |

#### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MESSAGES_API_URL` | `https://november7-730026606190.europe-west1.run.app/messages` | Aurora API endpoint |
| `HF_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model name |
| `GROQ_MODEL` | `llama3-8b-instruct` | Groq LLM model name |
| `TOP_K` | `5` | Number of messages to retrieve |
| `PORT` | `8000` | Server port number |

### Configuration File Example

`.env` file:
```env
# HuggingFace Configuration
HF_API_TOKEN=your_huggingface_token_here
HF_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-instruct

# Messages API Configuration
MESSAGES_API_URL=https://november7-730026606190.europe-west1.run.app/messages

# Retrieval Configuration
TOP_K=5

# Server Configuration
PORT=8000
```

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

## External APIs

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
- Handles 307 redirects automatically

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

## Performance Considerations

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

1. **Enable verbose logging**: Set log level to DEBUG in `app/main.py`
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

## Conclusion

This documentation provides a comprehensive overview of the Aurora QA System backend flow, covering:

- **Architecture**: Component structure and interactions
- **Startup**: Index building and initialization
- **Request Processing**: Complete question-to-answer pipeline
- **Components**: Detailed documentation of each module
- **Data Flow**: Visual and textual flow descriptions
- **Error Handling**: Error types, propagation, and fallbacks
- **Configuration**: Environment variables and settings
- **External APIs**: Integration details and usage
- **Performance**: Optimization considerations
- **Troubleshooting**: Common issues and solutions

For additional information, refer to:
- `README.md`: Setup and deployment instructions
- API Documentation: `http://localhost:8000/docs` (when running)
- Source code: Component docstrings and inline comments

