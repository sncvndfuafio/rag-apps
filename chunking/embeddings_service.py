# FIX: Changed from langchain_openai to langchain_community for HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingsService:
    def __init__(self):
        # FIX: Using 'paraphrase-MiniLM-L3-v2' model
        # The dimension for 'paraphrase-MiniLM-L3-v2' is also 384
        self.embeddings_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")

    def get_embeddings_model(self):
        return self.embeddings_model

embeddings_service = EmbeddingsService()