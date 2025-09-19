import uuid
from io import BytesIO
from typing import List, Tuple
import os

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rapidocr_onnxruntime import RapidOCR
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np

# NEW: Import HTTPException
from fastapi import HTTPException # <--- ADD THIS LINE

class DataIngestionService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.ocr_engine = RapidOCR()

    def extract_text_from_pdf(self, file_content: BytesIO) -> List[Tuple[str, int]]:
        """
        Extracts text from a PDF file, attempting OCR with RapidOCR if initial extraction yields little text.
        Returns a list of (text_content, page_number) tuples.
        """
        reader = PdfReader(file_content)
        all_page_texts_with_nums = []

        for page_num, page in enumerate(reader.pages):
            current_page_text = ""
            
            # Try standard text extraction
            text_from_pypdf = page.extract_text()
            if text_from_pypdf:
                current_page_text = text_from_pypdf
            
            # If pypdf extracts very little text, attempt OCR for this specific page
            if not current_page_text.strip() or len(current_page_text.strip()) < 50:
                print(f"Low text extracted by pypdf for page {page_num + 1}. Attempting OCR with RapidOCR...")
                
                # Reset file_content stream to the beginning for pdf2image
                file_content.seek(0)
                try:
                    # Convert only the current page to an image
                    # Use dpi=300 for better OCR quality if needed, default is 200
                    images = convert_from_bytes(file_content.read(), first_page=page_num + 1, last_page=page_num + 1, dpi=300)
                    if images:
                        pil_image = images[0]
                        
                        # FIX: Convert PIL Image to NumPy array (RapidOCR's preferred format)
                        # Ensure the image is in a common format like RGB if not already
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        numpy_image = np.array(pil_image)
                        
                        result, _ = self.ocr_engine(numpy_image) # Pass the NumPy array
                        if result:
                            # result is a list of lists, each sublist contains bbox, text, score
                            # We want to join the text parts
                            page_text_from_ocr = "\n".join([line[1] for line in result])
                            current_page_text = page_text_from_ocr
                            print(f"RapidOCR'd page {page_num + 1}")
                        else:
                            print(f"RapidOCR failed to extract text from page {page_num + 1}.")
                    else:
                        print(f"pdf2image failed to convert page {page_num + 1} to image.")
                except Exception as e:
                    print(f"OCR failed for page {page_num + 1} with error: {e}. Ensure Poppler is in PATH.")
                    current_page_text = "" # Fallback if OCR fails for this page

            all_page_texts_with_nums.append((current_page_text, page_num + 1))
            
        return all_page_texts_with_nums

    def chunk_text(self, page_texts_with_nums: List[Tuple[str, int]], file_id: str) -> List[Document]:
        """Chunks the extracted text into LangChain Documents with page number metadata."""
        all_documents = []
        for text_content, page_num in page_texts_with_nums:
            if not text_content.strip():
                continue # Skip empty pages

            chunks = self.text_splitter.create_documents([text_content])
            for i, chunk in enumerate(chunks):
                # Add file_id, chunk_id, and page_number to metadata
                chunk.metadata = {
                    "file_id": file_id,
                    "chunk_id": f"{file_id}-{page_num}-{i}",
                    "page_number": page_num
                }
                all_documents.append(chunk)
        
        if not all_documents:
            print(f"No text to chunk for file_id: {file_id}. Returning empty document list.")
        return all_documents

    def process_pdf(self, file_content: BytesIO) -> Tuple[str, List[Document]]:
        """
        Extracts text, chunks it, and returns the file_id and documents.
        Generates a unique file_id for the uploaded document.
        """
        file_id = str(uuid.uuid4())
        page_texts_with_nums = self.extract_text_from_pdf(file_content)
        documents = self.chunk_text(page_texts_with_nums, file_id)

        if not documents: # If no documents were created, raise an HTTPException
            raise HTTPException(status_code=400, detail="Could not extract any meaningful text or create chunks from the PDF.")

        return file_id, documents

data_ingestion_service = DataIngestionService()