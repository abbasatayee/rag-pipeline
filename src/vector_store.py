"""
Vector Store Module
Handles embedding creation and vector database management
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
from typing import List
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec


class VectorStore:
    """Manage vector store for document embeddings using Pinecone"""
    
    def __init__(
        self,
        index_name: str = "ancient-egypt-rag",
        use_openai: bool = True,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize vector store
        
        Args:
            index_name: Name of the Pinecone index
            use_openai: Whether to use OpenAI embeddings (requires API key)
            embedding_model: Name of the embedding model to use
        """
        self.index_name = index_name
        self.use_openai = use_openai
        self.embedding_model = embedding_model
        
        # Initialize Pinecone client
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        self.pinecone_client = PineconeClient(api_key=pinecone_api_key)
        
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
    
    def _get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        # Get dimension by embedding a test string
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)
    
    def delete_index(self):
        """Delete the Pinecone index"""
        existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
        if self.index_name in existing_indexes:
            print(f"Deleting existing index: {self.index_name}")
            self.pinecone_client.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted successfully")
    
    def create_vector_store(
        self,
        documents: List[Document],
        dimension: int = None,
        metric: str = "cosine",
        rebuild: bool = False
    ) -> Pinecone:
        """
        Create vector store from documents
        
        Args:
            documents: List of document chunks
            dimension: Dimension of the embeddings (auto-detected if None)
            metric: Distance metric for Pinecone (cosine, euclidean, dotproduct)
            rebuild: Whether to delete and recreate the index if it exists
            
        Returns:
            Pinecone vector store instance
        """
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Auto-detect dimension if not provided
        if dimension is None:
            dimension = self._get_embedding_dimension()
            print(f"Auto-detected embedding dimension: {dimension}")
        
        # Check if index exists
        existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
        
        if self.index_name in existing_indexes:
            if rebuild:
                self.delete_index()
            else:
                print(f"Using existing index: {self.index_name}")
        
        # Create index if it doesn't exist
        if self.index_name not in existing_indexes or rebuild:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{self.index_name}' created successfully")
        
        # Create vector store from documents
        vectorstore = Pinecone.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        
        print(f"Vector store created and saved to Pinecone index: {self.index_name}")
        return vectorstore
    
    def load_vector_store(self) -> Pinecone:
        """
        Load existing vector store
        
        Returns:
            Pinecone vector store instance
        """
        # Check if index exists
        existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
        
        if self.index_name not in existing_indexes:
            raise FileNotFoundError(
                f"Pinecone index '{self.index_name}' not found. "
                "Please create it first by calling create_vector_store()."
            )
        
        vectorstore = Pinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        print(f"Loaded vector store from Pinecone index: {self.index_name}")
        return vectorstore

