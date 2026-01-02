# Quick Start Guide

Get your RAG system up and running in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Your API Key

Create a `.env` file:

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

Replace `your_key_here` with your actual OpenAI API key from https://platform.openai.com/api-keys

## Step 3: Run the System

Place your PDF file in the project directory, then run:

```bash
python main.py --pdf your_document.pdf
```

The system will:
1. Process your PDF and create embeddings
2. Enter interactive mode where you can ask questions

## Example Session

```bash
$ python main.py --pdf ancient_egypt.pdf

Loading PDF from: ancient_egypt.pdf
Loaded 50 pages
Splitting into chunks of size 1000...
Created 234 chunks
Creating vector store with 234 documents...
Vector store saved to: ./chroma_db

✓ RAG system setup complete!

============================================================
RAG System Ready! Ask questions about your documents.
Type 'exit' or 'quit' to stop.
============================================================

Question: What were the main achievements of ancient Egypt?
Answer: [AI-generated answer based on your PDF]
```

## Next Steps

- Try different questions about your document
- Use `--rebuild` flag if you update your PDF
- Use `--query "your question"` for single queries without interactive mode

That's it! You're ready to query your PDF documents.

