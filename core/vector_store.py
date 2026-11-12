
"""Create and query a separate Chroma collection for each video."""

from functools import lru_cache
import re

import chromadb

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def validate_collection_name(collection_name):

    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        collection_name.strip(),
    )

    if not 3 <= len(cleaned) <= 60:
        raise ValueError(
            "Collection name must contain 3 to 60 valid characters."
        )

    return cleaned


def build_vector_store(
    transcript,
    collection_name,
    video_id=None,
):

    if not transcript or not transcript.strip():
        raise ValueError("Cannot index an empty transcript.")

    collection_name = validate_collection_name(collection_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(transcript)

    if not chunks:
        raise ValueError("No text chunks were created.")

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_index": index,
                "video_id": video_id or collection_name,
                "collection_name": collection_name,
            },
        )
        for index, chunk in enumerate(chunks)
    ]

    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )


def load_vector_store(collection_name):

    collection_name = validate_collection_name(collection_name)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store, k=4):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def delete_vector_collection(collection_name):

    collection_name = validate_collection_name(collection_name)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    try:
        client.delete_collection(name=collection_name)

    except Exception as exc:

        message = str(exc).lower()

        if (
            "does not exist" not in message
            and "not found" not in message
        ):
            raise
