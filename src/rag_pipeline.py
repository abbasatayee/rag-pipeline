"""
RAG Pipeline Module
Implements the Retrieval-Augmented Generation pipeline
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma, FAISS
from typing import List, Optional
import os


class RAGPipeline:
    """RAG pipeline for question answering"""
    
    def __init__(
        self,
        vectorstore: Chroma | FAISS,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 4
    ):
        """
        Initialize RAG pipeline
        
        Args:
            vectorstore: Vector store instance
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation
            top_k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        
        # Initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            
Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so.
Be accurate, concise, and cite specific details from the context when possible.

Context:
{context}

Question: {question}

Answer:"""),
        ])
        
        # Create the RAG chain
        self.chain = self._create_chain()
    
    def _create_chain(self):
        """Create the RAG chain"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _format_docs(self, docs: List) -> str:
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, question: str) -> str:
        """
        Query the RAG pipeline
        
        Args:
            question: User's question
            
        Returns:
            Answer from the RAG pipeline
        """
        return self.chain.invoke(question)
    
    def query_with_sources(self, question: str) -> dict:
        """
        Query the RAG pipeline and return answer with sources
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        
        # Get answer
        answer = self.query(question)
        
        # Extract sources
        sources = []
        for doc in docs:
            source_info = {
                "content": doc.page_content[:200] + "...",  # First 200 chars
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources
        }

