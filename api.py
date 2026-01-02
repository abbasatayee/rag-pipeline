"""
FastAPI Chat API for RAG System
Provides a chat-like interface for interacting with the RAG system
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Global variables to store initialized components
vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global vector_store, rag_pipeline
    
    try:
        # Check for required environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            print("⚠ WARNING: PINECONE_API_KEY not found in environment variables")
            print("   The API will start but chat functionality will not work.")
            print("   Please set PINECONE_API_KEY in your .env file.")
            yield
            return
        
        # Initialize vector store
        index_name = os.getenv("PINECONE_INDEX_NAME", "ancient-egypt-rag")
        use_openai_embeddings = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
        
        try:
            vector_store = VectorStore(
                index_name=index_name,
                use_openai=use_openai_embeddings
            )
            
            # Try to load existing vector store
            try:
                vectorstore = vector_store.load_vector_store()
                print(f"✓ Loaded existing vector store: {index_name}")
            except FileNotFoundError:
                print(f"⚠ Vector store '{index_name}' not found. Please create it first using main.py")
                print(f"   Run: python main.py --pdf path/to/your/document.pdf")
                vectorstore = None
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "Unauthorized" in error_msg or "Invalid API Key" in error_msg:
                    print("⚠ ERROR: Invalid Pinecone API Key")
                    print("   Please check your PINECONE_API_KEY in the .env file")
                    print("   Get your API key from: https://app.pinecone.io/")
                else:
                    print(f"⚠ Error loading vector store: {error_msg}")
                vectorstore = None
            
            # Initialize RAG pipeline if vector store is available
            if vectorstore:
                use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
                local_llm_base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
                local_llm_model = os.getenv("LOCAL_LLM_MODEL", "local-model")
                
                try:
                    rag_pipeline = RAGPipeline(
                        vectorstore,
                        top_k=4,
                        use_local_llm=use_local_llm,
                        local_llm_base_url=local_llm_base_url,
                        model_name=local_llm_model if use_local_llm else None
                    )
                    print("✓ RAG pipeline initialized")
                except Exception as e:
                    print(f"⚠ Error initializing RAG pipeline: {str(e)}")
                    print("   Chat functionality will not be available")
            else:
                print("⚠ RAG pipeline not initialized - vector store not available")
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg or "Invalid API Key" in error_msg:
                print("⚠ ERROR: Invalid Pinecone API Key")
                print("   Please check your PINECONE_API_KEY in the .env file")
                print("   Get your API key from: https://app.pinecone.io/")
            else:
                print(f"⚠ Error during startup: {error_msg}")
            print("   The API will start but chat functionality will not work.")
    
    except Exception as e:
        print(f"⚠ Unexpected error during startup: {str(e)}")
        print("   The API will start but chat functionality may not work.")
    
    yield
    
    # Shutdown (if needed)
    # Cleanup code can go here


app = FastAPI(title="RAG Chat API", version="1.0.0", lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class ChatMessage(BaseModel):
    """Chat message model"""
    message: str
    use_local_llm: bool = False


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[dict]
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vector_store_ready: bool
    rag_pipeline_ready: bool




@app.get("/")
async def root():
    """Root endpoint - serve the chat interface"""
    static_file = os.path.join(static_dir, "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {
        "message": "RAG Chat API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        vector_store_ready=vector_store is not None,
        rag_pipeline_ready=rag_pipeline is not None
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Chat endpoint - send a message and get a response from the RAG system
    
    Args:
        message: Chat message with optional use_local_llm flag
        
    Returns:
        Chat response with answer and sources
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please ensure the vector store is set up."
        )
    
    if not message.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )
    
    try:
        # If use_local_llm is specified in the request, create a new pipeline instance
        if message.use_local_llm:
            local_llm_base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
            local_llm_model = os.getenv("LOCAL_LLM_MODEL", "local-model")
            
            # Get the vectorstore from the vector_store
            vectorstore = vector_store.load_vector_store()
            
            # Create temporary pipeline with local LLM
            temp_rag = RAGPipeline(
                vectorstore,
                top_k=4,
                use_local_llm=True,
                local_llm_base_url=local_llm_base_url,
                model_name=local_llm_model
            )
            result = temp_rag.query_with_sources(message.message)
        else:
            # Use the default pipeline
            result = rag_pipeline.query_with_sources(message.message)
        
        return ChatResponse(
            answer=result['answer'],
            sources=result['sources'],
            message=message.message
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.post("/chat/simple")
async def chat_simple(message: str):
    """
    Simple chat endpoint - just send a string message
    
    Args:
        message: The chat message as a string
        
    Returns:
        Simple response with just the answer
    """
    chat_message = ChatMessage(message=message)
    response = await chat(chat_message)
    return {"answer": response.answer}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

