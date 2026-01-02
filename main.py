"""
Main RAG Application
Entry point for the RAG system
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline


def setup_rag(pdf_path: str, rebuild: bool = False):
    """
    Set up the RAG system by processing PDFs and creating vector store
    
    Args:
        pdf_path: Path to PDF file or directory containing PDFs
        rebuild: Whether to rebuild the vector store even if it exists
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    vector_store = VectorStore(
        index_name="ancient-egypt-rag",
        use_openai=False
    )
    
    # Check if vector store already exists
    if not rebuild:
        try:
            vectorstore = vector_store.load_vector_store()
            print("Vector store already exists. Use --rebuild to recreate it.")
            return vectorstore
        except FileNotFoundError:
            pass  # Index doesn't exist, will create it
    
    # Load PDF(s)
    pdf_path_obj = Path(pdf_path)
    if pdf_path_obj.is_file():
        documents = pdf_processor.load_pdf(pdf_path)
    elif pdf_path_obj.is_dir():
        pdf_files = list(pdf_path_obj.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {pdf_path}")
        documents = pdf_processor.load_multiple_pdfs([str(f) for f in pdf_files])
    else:
        raise ValueError(f"Invalid PDF path: {pdf_path}")
    
    # Create vector store
    vectorstore = vector_store.create_vector_store(documents, rebuild=rebuild)
    
    print("\n✓ RAG system setup complete!")
    return vectorstore


def interactive_query(vectorstore, use_local_llm: bool = False):
    """
    Interactive query interface
    
    Args:
        vectorstore: Vector store instance
        use_local_llm: Whether to use local LM Studio
    """
    # Get local LLM settings from environment
    local_llm_base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    local_llm_model = os.getenv("LOCAL_LLM_MODEL", "local-model")
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        vectorstore, 
        top_k=4,
        use_local_llm=use_local_llm,
        local_llm_base_url=local_llm_base_url,
        model_name=local_llm_model if use_local_llm else None
    )
    
    print("\n" + "="*60)
    print("RAG System Ready! Ask questions about your documents.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            print("\nThinking...")
            result = rag.query_with_sources(question)
            
            print("\n" + "-"*60)
            print("Answer:")
            print(result['answer'])
            print("\n" + "-"*60)
            print(f"\nRetrieved {len(result['sources'])} relevant chunks.")
            print("-"*60 + "\n")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            if use_local_llm:
                print("Please check that LM Studio is running and accessible.\n")
            else:
                print("Please check your OpenAI API key and try again.\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG System for PDF Documents")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store even if it exists"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (non-interactive mode)"
    )
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Use local LM Studio instead of OpenAI"
    )
    
    args = parser.parse_args()
    
    # Setup RAG system
    try:
        vectorstore = setup_rag(args.pdf, rebuild=args.rebuild)
    except Exception as e:
        print(f"Error setting up RAG system: {str(e)}")
        return
    
    # Get local LLM settings from environment
    local_llm_base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    local_llm_model = os.getenv("LOCAL_LLM_MODEL", "local-model")
    
    # Query mode
    if args.query:
        # Single query mode
        rag = RAGPipeline(
            vectorstore, 
            top_k=4,
            use_local_llm=args.local_llm,
            local_llm_base_url=local_llm_base_url,
            model_name=local_llm_model if args.local_llm else None
        )
        try:
            result = rag.query_with_sources(args.query)
            print("\nQuestion:", args.query)
            print("\nAnswer:", result['answer'])
            print(f"\nRetrieved {len(result['sources'])} relevant chunks.")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        # Interactive mode
        interactive_query(vectorstore, use_local_llm=args.local_llm)


if __name__ == "__main__":
    main()

