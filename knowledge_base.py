# knowledge_base.py
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from datetime import datetime
import pickle
from typing import List, Dict, Any

class GeneticsKnowledgeBase:
    def __init__(self, kb_path="./knowledge_base"):
        """Initialize genetics knowledge base with vector storage"""
        self.kb_path = kb_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Create knowledge base directory
        os.makedirs(kb_path, exist_ok=True)
        
        # Load existing knowledge base
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load existing knowledge base from disk"""
        index_path = os.path.join(self.kb_path, "faiss_index.bin")
        docs_path = os.path.join(self.kb_path, "documents.json")
        meta_path = os.path.join(self.kb_path, "metadata.pkl")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        if os.path.exists(docs_path):
            with open(docs_path, 'r') as f:
                self.documents = json.load(f)
        
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
    
    def _save_knowledge_base(self):
        """Save knowledge base to disk"""
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.kb_path, "faiss_index.bin"))
        
        with open(os.path.join(self.kb_path, "documents.json"), 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        with open(os.path.join(self.kb_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def add_document(self, text: str, source: str = "user", 
                    metadata: Dict[str, Any] = None):
        """Add a document to the knowledge base"""
        doc_id = len(self.documents)
        
        # Store document
        self.documents.append(text)
        
        # Store metadata
        meta = {
            'id': doc_id,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            ** (metadata or {})
        }
        self.metadata.append(meta)
        
        # Update vector index
        embedding = self.model.encode([text])
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embedding.astype('float32'))
        
        # Save updates
        self._save_knowledge_base()
        
        return doc_id
    
    def add_documents_batch(self, documents: List[str], sources: List[str] = None,
                           metadata_list: List[Dict] = None):
        """Add multiple documents at once"""
        if sources is None:
            sources = ["batch"] * len(documents)
        
        if metadata_list is None:
            metadata_list = [{}] * len(documents)
        
        for doc, source, meta in zip(documents, sources, metadata_list):
            self.add_document(doc, source, meta)
    
    def search(self, query: str, k: int = 5, threshold: float = 0.7):
        """Search for relevant documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k, len(self.documents))
        )
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                similarity = 1 - dist  # Convert distance to similarity
                
                if similarity >= threshold:
                    results.append({
                        'text': self.documents[idx],
                        'similarity': float(similarity),
                        'metadata': self.metadata[idx],
                        'id': int(idx)
                    })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:k]
    
    def load_from_directory(self, directory_path: str):
        """Load documents from a directory"""
        supported_extensions = ['.txt', '.json', '.md', '.pdf']
        
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                
                if ext == '.txt':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.add_document(content, source=filename)
                
                elif ext == '.json':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict):
                        # Extract all text content
                        texts = self._extract_text_from_dict(data)
                        for text in texts:
                            self.add_document(text, source=filename)
                    
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                self.add_document(item, source=filename)
                            elif isinstance(item, dict):
                                text = json.dumps(item)
                                self.add_document(text, source=filename)
    
    def _extract_text_from_dict(self, data: Dict, current_key: str = "") -> List[str]:
        """Recursively extract all text from a dictionary"""
        texts = []
        
        for key, value in data.items():
            if isinstance(value, str):
                texts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                texts.extend(self._extract_text_from_dict(value, key))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        texts.append(f"{key}: {item}")
                    elif isinstance(item, dict):
                        texts.extend(self._extract_text_from_dict(item, key))
        
        return texts
    
    def get_context_for_query(self, query: str, max_chars: int = 2000) -> str:
        """Get relevant context for a query"""
        results = self.search(query, k=3)
        
        if not results:
            return ""
        
        # Build context from top results
        context_parts = []
        total_chars = 0
        
        for result in results:
            text = result['text']
            if total_chars + len(text) <= max_chars:
                context_parts.append(f"[Source: {result['metadata'].get('source', 'Unknown')}]")
                context_parts.append(text)
                total_chars += len(text) + 50  # +50 for source tag
        
        return "\n\n".join(context_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'sources': list(set(m.get('source', 'Unknown') for m in self.metadata)),
            'last_updated': max(m.get('timestamp', '') for m in self.metadata) if self.metadata else None
        }
