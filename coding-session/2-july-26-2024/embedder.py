from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Optional
import streamlit as st

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


class Embedder(ABC, BaseModel):

    @abstractmethod
    def embed_query(self, text): ...

    @abstractmethod
    def embed_documents(self, documents, batch_size): ...

    def get_embedding_size(self):
        sample_text = "This is a sample text."
        sample_embedding = self.embedder.embed_query(sample_text)
        return len(sample_embedding)


class NVIDIAEmbedders(Embedder):
    name: str
    type: str
    embedder: Optional[Any] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = NVIDIAEmbeddings(model=self.name, model_type=self.type, nvidia_api_key=st.secrets["NVIDIA_API_KEY"])

    def embed_query(self, text):
        return self.embedder.embed_query(text)

    def embed_documents(self, documents, batch_size=10):
        output = []
        batch_documents = []
        for i, doc in enumerate(documents):
            batch_documents.append(doc)
            if len(batch_documents) == batch_size:
                output.extend(self.embedder.embed_documents(batch_documents))
                batch_documents = []
        else:
            if len(batch_documents) > 0:
                output.extend(self.embedder.embed_documents(batch_documents))
        return output
