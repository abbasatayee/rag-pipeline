"""
Example usage script for the RAG system
This demonstrates how to use the RAG system programmatically
"""

import os
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

def example_usage():
    """Example of how to use the RAG system"""
    
    # Step 1: Process PDF
    print("Step 1: Processing PDF...")
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    documents = pdf_processor.load_pdf("data/ancient-egypt.pdf")  # Replace with your PDF path
    # Step 2: Create vector store
    print("\nStep 2: Creating vector store...")
    vector_store = VectorStore(index_name="ancient-egypt-rag", use_openai=False)
    vectorstore = vector_store.create_vector_store(documents)
    
    # Step 3: Initialize RAG pipeline
    print("\nStep 3: Initializing RAG pipeline...")
    
    # Option 1: Use OpenAI (default)
    # rag = RAGPipeline(vectorstore, top_k=4)
    
    # Option 2: Use local LM Studio
    local_llm_base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    local_llm_model = os.getenv("LOCAL_LLM_MODEL", "local-model")
    rag = RAGPipeline(
        vectorstore, 
        top_k=4,
        use_local_llm=True,
        local_llm_base_url=local_llm_base_url,
        model_name=local_llm_model
    )
    
    result = rag.query_with_sources("What is the main topic of this document?")
    print(result['answer'])
    print(result['sources'])

if __name__ == "__main__":
    example_usage()

