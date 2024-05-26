from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def get_repo_file_contents() -> List[str]:
    """
    Gets all files by allowed file extensions and ignores excluded directories
    """
    loader = DirectoryLoader("./", glob="*.*")
    raw_documents = loader.load()

    return raw_documents


def get_split_documents(raw_documents: List[str]) -> List[str]:
    """
    Chunks codebase documents to be used in embeddings in a vector store
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(raw_documents)

    return split_documents


def embed_documents(index_name: str) -> None:
    """
    Embeds chunked documents in Pinecone's vector store after creating a new index
    """
    try:
        delete_embeddings_for_codebase(index_name)
    except:
        pass

    raw_documents = get_repo_file_contents()
    split_documents = get_split_documents(raw_documents)

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large"
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    PineconeVectorStore.from_documents(
        documents=split_documents,
        embedding=embeddings,
        index_name=index_name,
    )


def delete_embeddings_for_codebase(index_name: str) -> str:
    """
    Deletes an index from the vector database account
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc.delete_index(index_name)
