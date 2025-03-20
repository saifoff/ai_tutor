import os
from typing import List, Dict
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        documents = []
        
        # Load different types of documents with proper encoding
        loaders = {
            ".txt": DirectoryLoader(
                curriculum_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            ),
            ".md": DirectoryLoader(
                curriculum_dir,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader
            )
        }
        
        for ext, loader in loaders.items():
            try:
                docs = loader.load()
                if docs:  # Only add if documents were loaded
                    documents.extend(docs)
                    logger.info(f"Successfully loaded {len(docs)} {ext} files from {curriculum_dir}")
                else:
                    logger.warning(f"No {ext} files found in {curriculum_dir}")
            except Exception as e:
                logger.error(f"Error loading {ext} files from {curriculum_dir}: {str(e)}")
        
        if not documents:
            logger.warning(f"No documents were loaded from {curriculum_dir}")
            return
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        if not texts:
            logger.warning("No text chunks were created from the documents")
            return
            
        # Create vector store
        try:
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            logger.info(f"Successfully created vector store with {len(texts)} chunks")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def retrieve_relevant_content(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant curriculum content for a query"""
        if self.vector_store is None:
            logger.warning("Vector store not initialized. No content available for retrieval.")
            return []
        
        try:
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
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)}")
            return []
    
    def save_vector_store(self, path: str):
        """Save the vector store to disk"""
        if self.vector_store is not None:
            try:
                self.vector_store.save_local(path)
                logger.info(f"Successfully saved vector store to {path}")
            except Exception as e:
                logger.error(f"Error saving vector store: {str(e)}")
    
    def load_vector_store(self, path: str):
        """Load the vector store from disk"""
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings)
            logger.info(f"Successfully loaded vector store from {path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

class MultiLanguageRetriever:
    def __init__(self, supported_languages: List[str]):
        self.retrievers = {}
        for lang in supported_languages:
            self.retrievers[lang] = CurriculumRetriever()
    
    def load_curriculum(self, curriculum_dir: str, language: str):
        """Load curriculum for a specific language"""
        if language not in self.retrievers:
            logger.error(f"Language {language} not supported")
            raise ValueError(f"Language {language} not supported")
        
        lang_dir = os.path.join(curriculum_dir, language)
        if not os.path.exists(lang_dir):
            logger.warning(f"Directory {lang_dir} does not exist")
            return
            
        self.retrievers[language].load_curriculum(lang_dir)
    
    def retrieve_relevant_content(self, query: str, language: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant content for a specific language"""
        if language not in self.retrievers:
            logger.error(f"Language {language} not supported")
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