import os
from typing import List, Dict
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader

class CurriculumRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def load_curriculum(self, curriculum_dir: str):
        """Load curriculum documents from directory"""
        # Load different types of documents
        loaders = {
            ".txt": DirectoryLoader(curriculum_dir, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(curriculum_dir, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
        }
        
        documents = []
        for ext, loader in loaders.items():
            try:
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {ext} files: {e}")
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
    def retrieve_relevant_content(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant curriculum content for a query"""
        if self.vector_store is None:
            raise ValueError("Curriculum not loaded. Call load_curriculum first.")
        
        # Get relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0)
            })
        
        return results
    
    def save_vector_store(self, path: str):
        """Save the vector store to disk"""
        if self.vector_store is not None:
            self.vector_store.save_local(path)
    
    def load_vector_store(self, path: str):
        """Load the vector store from disk"""
        self.vector_store = FAISS.load_local(path, self.embeddings)

class MultiLanguageRetriever:
    def __init__(self, supported_languages: List[str]):
        self.retrievers = {}
        for lang in supported_languages:
            self.retrievers[lang] = CurriculumRetriever()
    
    def load_curriculum(self, curriculum_dir: str, language: str):
        """Load curriculum for a specific language"""
        if language not in self.retrievers:
            raise ValueError(f"Language {language} not supported")
        
        lang_dir = os.path.join(curriculum_dir, language)
        self.retrievers[language].load_curriculum(lang_dir)
    
    def retrieve_relevant_content(self, query: str, language: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant content for a specific language"""
        if language not in self.retrievers:
            raise ValueError(f"Language {language} not supported")
        
        return self.retrievers[language].retrieve_relevant_content(query, k)
    
    def save_vector_stores(self, base_path: str):
        """Save all vector stores"""
        for lang, retriever in self.retrievers.items():
            path = os.path.join(base_path, lang)
            retriever.save_vector_store(path)
    
    def load_vector_stores(self, base_path: str):
        """Load all vector stores"""
        for lang, retriever in self.retrievers.items():
            path = os.path.join(base_path, lang)
            retriever.load_vector_store(path) 