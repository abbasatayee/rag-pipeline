# RAG System for PDF Documents

A Retrieval-Augmented Generation (RAG) system that allows you to query PDF documents and get accurate answers based on your knowledge source.

## Features

- 📄 **PDF Processing**: Load and chunk PDF documents efficiently
- 🔍 **Semantic Search**: Find relevant information using vector embeddings
- 💬 **Question Answering**: Get accurate answers based on your documents
- 🚀 **Easy to Use**: Simple command-line interface
- 💾 **Persistent Storage**: Vector store is saved for fast subsequent queries

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
# Required: Pinecone API key for vector storage
PINECONE_API_KEY=your_pinecone_api_key_here

# Option 1: Use OpenAI for embeddings and LLM
OPENAI_API_KEY=your_openai_api_key_here

# Option 2: Use local LM Studio for LLM (optional)
LOCAL_LLM_BASE_URL=http://localhost:1234/v1
LOCAL_LLM_MODEL=local-model
```

Get your API keys from:
- [Pinecone Console](https://app.pinecone.io/) - for vector storage
- [OpenAI Platform](https://platform.openai.com/api-keys) - for embeddings and LLM (if not using local)

### 3. Prepare Your PDF Files

Place your PDF file(s) in the project directory or note the path to your PDF file.

## Usage

### Initial Setup (Process PDF and Create Vector Store)

```bash
python main.py --pdf path/to/your/document.pdf
```

Or for multiple PDFs in a directory:

```bash
python main.py --pdf path/to/pdf/directory/
```

### Interactive Query Mode

After initial setup, you can query interactively:

```bash
python main.py --pdf path/to/your/document.pdf
```

Then type your questions when prompted.

### Single Query Mode

For a single question:

```bash
python main.py --pdf path/to/your/document.pdf --query "Your question here"
```

### Using Local LM Studio

To use your local LM Studio instead of OpenAI:

```bash
# Interactive mode with local LLM
python main.py --pdf path/to/your/document.pdf --local-llm

# Single query with local LLM
python main.py --pdf path/to/your/document.pdf --query "Your question" --local-llm
```

Make sure LM Studio is running and accessible at the configured URL (default: `http://localhost:1234/v1`).

### Rebuild Vector Store

To rebuild the vector store (useful after updating PDFs):

```bash
python main.py --pdf path/to/your/document.pdf --rebuild
```

## Example

```bash
# First time: Process PDF and create vector store
python main.py --pdf ancient_egypt_history.pdf

# Then you'll enter interactive mode:
# Question: What were the main achievements of ancient Egypt?
# Answer: [AI-generated answer based on your PDF]
```

## Project Structure

```
.
├── src/
│   ├── __init__.py         # Package initialization
│   ├── pdf_processor.py    # PDF loading and chunking
│   ├── vector_store.py     # Vector store management
│   └── rag_pipeline.py     # RAG pipeline implementation
├── main.py                 # Main application entry point
├── example_usage.py        # Example usage script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
└── README.md              # This file
```

## How It Works

1. **PDF Processing**: Your PDF is loaded and split into smaller chunks (1000 characters with 200 character overlap)

2. **Embedding Creation**: Each chunk is converted into a vector embedding using OpenAI's embedding model (or local embeddings)

3. **Vector Store**: Embeddings are stored in Pinecone for fast similarity search

4. **Query Processing**: When you ask a question:
   - Your question is converted to an embedding
   - Similar document chunks are retrieved
   - The retrieved context and your question are sent to the LLM
   - You get an accurate answer based on your documents

## Configuration

You can customize the system by modifying:

- **Chunk size**: Edit `chunk_size` and `chunk_overlap` in `src/pdf_processor.py`
- **Number of retrieved documents**: Edit `top_k` in `RAGPipeline` initialization
- **LLM model**: 
  - For OpenAI: Set `model_name` in `RAGPipeline` initialization
  - For local LM Studio: Set `LOCAL_LLM_MODEL` in `.env` or pass `model_name` parameter
- **Local LLM URL**: Set `LOCAL_LLM_BASE_URL` in `.env` (default: `http://localhost:1234/v1`)
- **Embedding model**: Modify `embedding_model` in `VectorStore` initialization
- **Pinecone index**: Change `index_name` in `VectorStore` initialization

## Troubleshooting

### "PINECONE_API_KEY not found"
- Make sure you've created a `.env` file with your Pinecone API key
- Get your key from [Pinecone Console](https://app.pinecone.io/)

### "OPENAI_API_KEY not found"
- This error appears if you're not using `--local-llm` flag
- Either add `OPENAI_API_KEY` to your `.env` file or use `--local-llm` flag
- Note: OpenAI API key is still required for embeddings even when using local LLM

### Local LM Studio connection issues
- Make sure LM Studio is running and a model is loaded
- Check that the server is accessible at the URL specified in `LOCAL_LLM_BASE_URL`
- Default URL is `http://localhost:1234/v1` - adjust if your LM Studio uses a different port
- Verify the model name matches what's loaded in LM Studio

### "PDF file not found"
- Verify the path to your PDF file is correct
- Use absolute paths if relative paths don't work

### Slow processing
- Large PDFs may take time to process initially
- Subsequent queries will be fast as the vector store is cached in Pinecone
- Local LLM responses may be slower than OpenAI depending on your hardware

## License

This project is open source and available for your use.
