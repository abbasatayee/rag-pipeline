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
    documents = pdf_processor.load_pdf("your_document.pdf")  # Replace with your PDF path
    
    # Step 2: Create vector store
    print("\nStep 2: Creating vector store...")
    vector_store = VectorStore(persist_directory="./chroma_db", use_openai=True)
    vectorstore = vector_store.create_vector_store(documents)
    
    # Step 3: Initialize RAG pipeline
    print("\nStep 3: Initializing RAG pipeline...")
    rag = RAGPipeline(vectorstore, top_k=4)
    
    # Step 4: Query the system
    print("\nStep 4: Querying the system...")
    questions = [
        "What is the main topic of this document?",
        "Summarize the key points.",
        # Add your questions here
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query_with_sources(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])} chunks retrieved")

if __name__ == "__main__":
    example_usage()

