"""
PDF Processing Module
Handles loading and chunking PDF documents for RAG
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from pathlib import Path


class PDFProcessor:
    """Process PDF files and split them into chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, pdf_path: str) -> List:
        """
        Load PDF file and split into chunks
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document chunks
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages")
        print(f"Splitting into chunks of size {self.chunk_size}...")
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def load_multiple_pdfs(self, pdf_paths: List[str]) -> List:
        """
        Load multiple PDF files and combine them
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        for pdf_path in pdf_paths:
            chunks = self.load_pdf(pdf_path)
            all_chunks.extend(chunks)
        
        return all_chunks

