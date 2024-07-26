from pydantic import BaseModel
from embedder import Embedder
from vectordb import VectorClient

class Retriever(BaseModel):

    embedder : Embedder
    vector_client : VectorClient
    search_limit : int = 4

    def get_relevant_docs(self, text, limit=None):
        if limit is None:
            limit = self.search_limit
        query_vector = self.embedder.embed_query(text)
        concatdocs, sources = self.vector_client.search([query_vector], limit)
        return concatdocs, sources