import os

class Config:
    """
    Configuration class to manage file paths and settings for PDF-based RAG.
    """

    # --- PDF Configuration ---
    # Directory where source PDF documents are located
    PDF_SOURCE_DIRECTORY: str = "data"

    # Directory where ChromaDB embeddings will be persisted
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma"

    # --- Embedding & Chunking Configuration ---
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
    CHUNK_SIZE: int = 2028
    CHUNK_OVERLAP: int = 250

    def __init__(self):
        # Ensure required directories exist
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

        print("Configuration loaded successfully.")
        print(f"PDF source directory : {self.PDF_SOURCE_DIRECTORY}")
        print(f"Chroma persist dir  : {self.CHROMA_PERSIST_DIRECTORY}")

# Global config instance
config = Config()
