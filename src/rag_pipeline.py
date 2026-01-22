"""
RAG Pipeline Module
Implements the Retrieval-Augmented Generation pipeline
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from typing import List, Optional
import os

# Use the new langchain-pinecone package
try:
    from langchain_pinecone import Pinecone
except ImportError:
    raise ImportError(
        "langchain-pinecone package is required. "
        "Install it with: pip install langchain-pinecone"
    )


class RAGPipeline:
    """RAG pipeline for question answering"""
    
    def __init__(
        self,
        vectorstore: Pinecone,
        model_name: str = None,
        temperature: float = 0.0,
        top_k: int = 4,
        use_local_llm: bool = False,
        local_llm_base_url: str = "http://localhost:1234/v1",
        local_llm_api_key: str = "lm-studio"
    ):
        """
        Initialize RAG pipeline
        
        Args:
            vectorstore: Vector store instance
            model_name: Name of the LLM model to use (defaults based on use_local_llm)
            temperature: Temperature for LLM generation
            top_k: Number of documents to retrieve
            use_local_llm: Whether to use local LM Studio instead of OpenAI
            local_llm_base_url: Base URL for local LM Studio (default: http://localhost:1234/v1)
            local_llm_api_key: API key for local LM Studio (default: "lm-studio")
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        
        # Set default model name if not provided
        if model_name is None:
            model_name = "local-model" if use_local_llm else "gpt-3.5-turbo"
        
        # Initialize LLM
        if use_local_llm:
            # Use local LM Studio
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                base_url=local_llm_base_url,
                api_key=local_llm_api_key
            )
            print(f"Using local LLM at {local_llm_base_url} with model: {model_name}")
        else:
            # Use OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please set it in your .env file or use use_local_llm=True."
                )
            
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key
            )
            print(f"Using OpenAI with model: {model_name}")
        
        print("✓ LLM initialized")
        print("✓ Vector store connected")
        print(f"✓ Top-k documents to retrieve: {self.top_k}")
        print("✓ Initializing RAG chain...")
        print("✓ Creating prompt template...")
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a retrieval-augmented assistant. Your job is to answer the user ONLY using the information provided in the CONTEXT section. If the answer is not in the CONTEXT, you must say you do not have enough information from the provided sources and ask for what is needed (a clarifying question) or explain what information would be required.

NON-NEGOTIABLE RULES
1) Use ONLY the CONTEXT to make factual claims. Do not use outside knowledge.
2) If the CONTEXT does not contain the answer, do NOT guess, do NOT invent details, and do NOT “fill in the blanks.”
3) If the user’s question is ambiguous, ask clarifying questions before answering.
4) If the user requests actions, policies, prices, legal/medical/financial advice, or anything high-stakes, you must be extra conservative and require explicit support from CONTEXT. If not present, refuse to speculate and ask for authoritative source material.
5) If the CONTEXT contains conflicting information, explicitly call it out, quote/point to the conflicting parts, and ask which source/timeframe to trust.

ANSWER FORMAT
- Start with a direct answer if supported by CONTEXT.
- Include citations for each key claim in the form (or whatever identifiers are available for each chunk).
- Keep answers concise and precise; prefer “based on the provided sources…” language.

CONTEXT:
{context}

USER QUESTION:
{question}

Answer:"""),
        ])
        print("✓ Prompt template created")
        print("✓ Creating RAG chain...")
        
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
        docs = self.vectorstore.similarity_search(question, k=self.top_k , threshold_on_relevance=True,score_threshold = 0.6)
        print(f"Retrieved {len(docs)} documents for the question.")
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

