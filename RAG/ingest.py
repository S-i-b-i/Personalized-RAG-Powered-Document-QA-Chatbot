import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from config import config


def load_pdf_content(pdf_directory: str):
    """
    Loads PDF documents from a specified directory and returns them as a list of pages.
    """
    print(f"Starting PDF document loading from '{pdf_directory}'...")

    if not os.path.exists(pdf_directory):
        print(f"Error: PDF directory '{pdf_directory}' not found.")
        return []

    all_pdf_docs = []

    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            print(f"Loading PDF: {filepath}")
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                all_pdf_docs.extend(pages)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    if not all_pdf_docs:
        print("No PDF documents loaded.")
    else:
        print(f"Loaded {len(all_pdf_docs)} PDF pages.")

    return all_pdf_docs


def ingest_pdf_documents(
    pdf_directory: str = "data",
    persist_directory: str = "docs/chroma"
):
    """
    Loads PDFs, splits text, generates embeddings, and stores them in Chroma DB.
    """
    print("\n--- Starting PDF ingestion process ---")

    # 1. Load PDFs
    documents = load_pdf_content(pdf_directory)
    if not documents:
        print("No documents found. Exiting.")
        return

    # 2. Split documents
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Embeddings
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    # 4. Store in Chroma
    print(f"Persisting Chroma DB at '{persist_directory}'...")
    os.makedirs(persist_directory, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    print("PDF ingestion completed successfully!")
    print(f"Vector DB stored at: {persist_directory}")
    print("--- Ingestion complete ---")


if __name__ == "__main__":
    ingest_pdf_documents(
        pdf_directory=config.PDF_SOURCE_DIRECTORY,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )
