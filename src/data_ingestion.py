import os
import yaml
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def ingest_data(config):
    data_dir = config['data']['raw_data_path']
    chunk_size = config['data']['chunk_size']
    chunk_overlap = config['data']['chunk_overlap']
    embedding_model_name = config['data']['embedding_model']
    persist_directory = config['data']['vector_db_path']

    print("Loading raw documents...")
    # Supporting standard PDF and text files for classical literature
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    text_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    
    docs = pdf_loader.load() + text_loader.load()
    print(f"Loaded {len(docs)} documents from {data_dir}.")

    if not docs:
        print("No documents found. Please ensure files are placed in the raw_data_path.")
        return

    print(f"Chunking texts (Token Size: {chunk_size}, Overlap: {chunk_overlap})...")
    # Using RecursiveCharacterTextSplitter as requested
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print(f"Initializing Embedding Model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Storing chunks in ChromaDB at {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Ingestion pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    ingest_data(config)
