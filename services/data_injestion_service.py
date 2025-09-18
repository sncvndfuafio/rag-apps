import uuid
from io import BytesIO
from typing import List, Tuple

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DataIngestionService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def extract_text_from_pdf(self, file_content: BytesIO) -> str:
        """Extracts text from a PDF file."""
        reader = PdfReader(file_content)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def chunk_text(self, text: str, file_id: str) -> List[Document]:
        """Chunks the extracted text into LangChain Documents."""
        chunks = self.text_splitter.create_documents([text])
        # Add source metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata = {"file_id": file_id, "chunk_id": f"{file_id}-{i}"}
        return chunks

    def process_pdf(self, file_content: BytesIO) -> Tuple[str, List[Document]]:
        """
        Extracts text, chunks it, and returns the file_id and documents.
        Generates a unique file_id for the uploaded document.
        """
        file_id = str(uuid.uuid4())
        extracted_text = self.extract_text_from_pdf(file_content)
        documents = self.chunk_text(extracted_text, file_id)
        return file_id, documents

data_ingestion_service = DataIngestionService()