# rag_app/api/routes_files.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from io import BytesIO
from services.data_injestion_service import data_ingestion_service
# FIX: Corrected import path for vectordb_service
from chunking.vectordb_service import vectordb_service
from typing import Dict, Any

router = APIRouter()

@router.post("/add_file", response_model=Dict[str, str])
async def add_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_content = BytesIO(await file.read())
    file_id, documents = data_ingestion_service.process_pdf(file_content)

    if not documents:
        raise HTTPException(status_code=400, detail="Could not extract text or create chunks from the PDF.")

    vectordb_service.add_documents(documents, file_id)
    return {"file_id": file_id, "message": "File uploaded and processed successfully!"}

@router.delete("/delete_file/{file_id}", response_model=Dict[str, str])
async def delete_file(file_id: str):
    try:
        response = vectordb_service.delete_documents_by_file_id(file_id)
        return {"message": f"File '{file_id}' and its embeddings deleted successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file '{file_id}': {str(e)}")

@router.put("/update_file/{file_id}", response_model=Dict[str, str])
async def update_file(file_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        vectordb_service.delete_documents_by_file_id(file_id)
    except Exception as e:
        print(f"Warning: Could not delete existing embeddings for file_id {file_id}. Proceeding to add new. Error: {e}")

    file_content = BytesIO(await file.read())
    extracted_text = data_ingestion_service.extract_text_from_pdf(file_content)
    documents = data_ingestion_service.chunk_text(extracted_text, file_id)

    if not documents:
        raise HTTPException(status_code=400, detail="Could not extract text or create chunks from the new PDF.")

    vectordb_service.add_documents(documents, file_id)
    return {"file_id": file_id, "message": f"File '{file_id}' updated successfully!"}