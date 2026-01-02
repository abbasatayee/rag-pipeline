# RAG Chat API

A FastAPI-based REST API for interacting with the RAG system through a chat interface.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your `.env` file is configured with:
```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=ancient-egypt-rag  # Optional, defaults to ancient-egypt-rag
USE_OPENAI_EMBEDDINGS=false  # Set to true if using OpenAI embeddings
OPENAI_API_KEY=your_openai_key  # Required if USE_OPENAI_EMBEDDINGS=true

# Optional: For local LLM
USE_LOCAL_LLM=false  # Set to true to use local LM Studio by default
LOCAL_LLM_BASE_URL=http://localhost:1234/v1
LOCAL_LLM_MODEL=local-model

# Optional: API configuration
API_PORT=8000  # Default port for the API server
```

3. Make sure your vector store is set up:
```bash
python main.py --pdf path/to/your/document.pdf
```

## Running the API

### Option 1: Using Python directly
```bash
python api.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### `GET /`
Serves the chat interface HTML page.

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "vector_store_ready": true,
  "rag_pipeline_ready": true
}
```

### `POST /chat`
Main chat endpoint for sending messages.

**Request Body:**
```json
{
  "message": "What is the main topic of the document?",
  "use_local_llm": false
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "content": "Document chunk content...",
      "metadata": {...}
    }
  ],
  "message": "What is the main topic of the document?"
}
```

### `POST /chat/simple`
Simplified chat endpoint that only returns the answer.

**Request Body:**
```json
"Your question here"
```

**Response:**
```json
{
  "answer": "The answer to your question..."
}
```

## Using the Chat Interface

1. Open http://localhost:8000 in your browser
2. Type your question in the input field
3. Optionally check "Use Local LLM" to use LM Studio instead of OpenAI
4. Click "Send" or press Enter
5. View the response with source citations

## API Usage Examples

### Using cURL

```bash
# Simple chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is ancient Egypt known for?",
    "use_local_llm": false
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What is ancient Egypt known for?",
        "use_local_llm": False
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Sources: {len(data['sources'])} chunks retrieved")
```

### Using JavaScript/Fetch

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'What is ancient Egypt known for?',
    use_local_llm: false
  })
});

const data = await response.json();
console.log('Answer:', data.answer);
console.log('Sources:', data.sources);
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (e.g., empty message)
- `503`: Service unavailable (vector store not initialized)
- `500`: Internal server error

Error responses include a `detail` field with the error message:

```json
{
  "detail": "Error message here"
}
```

## Development

To run with auto-reload during development:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment

For production, consider:

1. Using a production ASGI server like Gunicorn with Uvicorn workers:
```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. Setting up proper CORS origins instead of allowing all origins
3. Adding authentication/authorization
4. Using environment-specific configuration
5. Setting up logging and monitoring

