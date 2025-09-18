from fastapi import FastAPI
# FIX: Corrected import paths
from api import routes_chat, routes_files

app = FastAPI(
    title="Agentic RAG System for PDFs",
    description="A Retrieval-Augmented Generation (RAG) system with LangGraph, Pinecone, and Groq for PDF documents.",
    version="1.0.0",
)

app.include_router(routes_files.router, tags=["File Management"])
app.include_router(routes_chat.router, tags=["Chat"])

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Agentic RAG System! Visit /docs for API documentation."}