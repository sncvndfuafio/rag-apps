# Agentic RAG System with LangGraph, Pinecone, and OpenAI

This project implements a Retrieval-Augmented Generation (RAG) system specifically designed for PDF documents. It uses LangGraph for orchestration, Pinecone as the vector database, OpenAI embeddings for text representation, and an OpenAI LLM for query answering, all exposed via a FastAPI service.

## Features

-   **PDF Support:** Only PDF files are supported for document ingestion.
-   **Vector Database:** Pinecone is used to store and retrieve document embeddings.
-   **Embeddings:** OpenAI's text-embedding-ada-002 model is used for generating embeddings.
-   **LLM:** OpenAI's GPT models (e.g., `gpt-3.5-turbo`) are used for generating answers.
-   **Graph Orchestration:** LangGraph is employed to manage the RAG workflow.
-   **FastAPI:** A RESTful API provides endpoints for file management and chat.

## Directory Structure