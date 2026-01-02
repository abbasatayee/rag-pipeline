"""
Vector Store Module
Handles embedding creation and vector database management
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Union
import os
from pathlib import Path


class VectorStore:
    """Manage vector store for document embeddings"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        use_openai: bool = True,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist vector store
            use_openai: Whether to use OpenAI embeddings (requires API key)
            embedding_model: Name of the embedding model to use
        """
        self.persist_directory = persist_directory
        self.use_openai = use_openai
        self.embedding_model = embedding_model
        
        # Initialize embeddings
        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please set it or use use_openai=False for local embeddings."
                )
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key
            )
        else:
            # Use local embeddings (sentence-transformers)
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
    
    def create_vector_store(
        self,
        documents: List[Document],
        use_chroma: bool = True
    ) -> Union[Chroma, FAISS]:
        """
        Create vector store from documents
        
        Args:
            documents: List of document chunks
            use_chroma: Whether to use ChromaDB (True) or FAISS (False)
            
        Returns:
            Vector store instance
        """
        print(f"Creating vector store with {len(documents)} documents...")
        
        if use_chroma:
            # Remove existing database if it exists
            if Path(self.persist_directory).exists():
                import shutil
                shutil.rmtree(self.persist_directory)
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Vector store saved to: {self.persist_directory}")
        else:
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Save FAISS index
            vectorstore.save_local(self.persist_directory)
            print(f"FAISS index saved to: {self.persist_directory}")
        
        return vectorstore
    
    def load_vector_store(self, use_chroma: bool = True) -> Union[Chroma, FAISS]:
        """
        Load existing vector store
        
        Args:
            use_chroma: Whether to use ChromaDB (True) or FAISS (False)
            
        Returns:
            Vector store instance
        """
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(
                f"Vector store not found at: {self.persist_directory}. "
                "Please create it first."
            )
        
        if use_chroma:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            vectorstore = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        print(f"Loaded vector store from: {self.persist_directory}")
        return vectorstore

