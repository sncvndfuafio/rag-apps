from typing import List, Optional
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from core.config import config
from chunking.embeddings_service import embeddings_service

class VectorDBService:
    def __init__(self):
        self.pinecone = Pinecone(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENVIRONMENT)
        self.index_name = config.PINECONE_INDEX_NAME
        self.embeddings_model = embeddings_service.get_embeddings_model()
        self.vectorstore: Optional[PineconeVectorStore] = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=384, # Ensure this matches your HuggingFace embedding model
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": config.PINECONE_ENVIRONMENT}}
            )
        self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings_model)

    def add_documents(self, documents: List[Document], file_id: str):
        for doc in documents:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["file_id"] = file_id
        ids = self.vectorstore.add_documents(documents)
        print(f"Added {len(ids)} vectors for file_id: {file_id}")
        return ids

    def delete_documents_by_file_id(self, file_id: str):
        index = self.pinecone.Index(self.index_name)
        response = index.delete(filter={"file_id": file_id})
        print(f"Deleted vectors for file_id: {file_id}. Response: {response}")
        return response

    def get_retriever(self):
        """Returns a LangChain retriever for the Pinecone vector store."""
        if self.vectorstore is None:
            self._initialize_vectorstore()
        # FIX: Increase 'k' to retrieve more documents
        return self.vectorstore.as_retriever(search_kwargs={"k": 10}) # Retrieve 10 most relevant chunks

vectordb_service = VectorDBService()