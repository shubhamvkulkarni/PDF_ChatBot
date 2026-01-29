#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# core.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain.schema import Document  # may vary by langchain version; fallback is OK

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

def upload_pdf(src_path: str, dest_dir: str) -> str:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"PDF not found: {src_path}")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) != os.path.abspath(dest):
        with open(src_path, "rb") as fsrc, open(dest, "wb") as fdst:
            fdst.write(fsrc.read())
    return dest

def create_or_load_vector_store(pdf_path: str, vs_root: str, embeddings, reindex: bool = False):
    """
    Create or load a FAISS vector store for a single PDF.
    embeddings must implement embed_documents/embed_query (see embeddings.py)
    """
    os.makedirs(vs_root, exist_ok=True)
    name = os.path.basename(pdf_path)
    store_dir = os.path.join(vs_root, name + "_faiss")

    if reindex and os.path.exists(store_dir):
        import shutil
        shutil.rmtree(store_dir)

    if os.path.exists(store_dir):
        # load existing vector store (only if you trust the files)
        db = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)

        return db, store_dir

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        raise RuntimeError("No pages loaded from PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    try:
        db.save_local(store_dir)
    except Exception:
        # not fatal; continue using in-memory index
        pass
    return db, store_dir

# def retrieve_docs(db, query: str, k: int = 4) -> List[Document]:
def retrieve_docs(db, query: str, k: int = 4):
    # handle different langchain signatures gracefully
    try:
        return db.similarity_search(query, k=k)
    except TypeError:
        return db.similarity_search(query, k)
