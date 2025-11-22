import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json

class GeneticsKnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.knowledge_data = []
        self.dimension = 384
        
    def add_knowledge(self, text, source="user", metadata=None):
        """Add new genetics knowledge to vector database"""
        knowledge_item = {
            'text': text,
            'source': source,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        self.knowledge_data.append(knowledge_item)
        
        # Update vector index
        self._update_index()
    
    def _update_index(self):
        """Update FAISS index with new knowledge"""
        if not self.knowledge_data:
            return
            
        texts = [item['text'] for item in self.knowledge_data]
        embeddings = self.model.encode(texts)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings.astype('float32'))
    
    def search_similar(self, query, k=5):
        """Search for similar genetics knowledge"""
        if not self.knowledge_data:
            return []
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.knowledge_data):
                results.append({
                    'text': self.knowledge_data[idx]['text'],
                    'similarity': 1 - distances[0][i],
                    'metadata': self.knowledge_data[idx]['metadata']
                })
        
        return results
