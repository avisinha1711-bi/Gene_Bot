import requests
import json
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from datetime import datetime

class GeneticsChatBot:
    def __init__(self):
        self.knowledge_base = self.load_knowledge_base()
        self.vectorizer = TfidfVectorizer()
        self.user_feedback = []
        self.conversation_history = []
        
    def load_knowledge_base(self):
        """Load existing genetics knowledge"""
        try:
            with open('genetics_kb.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {"genes": {}, "diseases": {}, "research_papers": {}}
    
    def save_knowledge_base(self):
        """Save updated knowledge"""
        with open('genetics_kb.pkl', 'wb') as f:
            pickle.dump(self.knowledge_base, f)
    
    def fetch_pubmed_data(self, query):
        """Fetch latest genetics research from PubMed"""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': f"{query} genetics",
            'retmode': 'json',
            'retmax': 5
        }
        try:
            response = requests.get(base_url, params=params)
            return response.json()
        except Exception as e:
            print(f"PubMed API error: {e}")
            return None
    
    def learn_from_interaction(self, question, answer, user_feedback):
        """Learn from user interactions"""
        learning_data = {
            'question': question,
            'answer': answer,
            'feedback': user_feedback,
            'timestamp': datetime.now()
        }
        self.user_feedback.append(learning_data)
        
        # Update knowledge base based on feedback
        if user_feedback == "positive":
            self._reinforce_knowledge(question, answer)
        elif user_feedback == "negative":
            self._flag_knowledge_gap(question)
    
    def _reinforce_knowledge(self, question, answer):
        """Strengthen correct knowledge patterns"""
        # Add to frequently asked questions
        if 'faq' not in self.knowledge_base:
            self.knowledge_base['faq'] = {}
        self.knowledge_base['faq'][question] = answer
        self.save_knowledge_base()
    
    def _flag_knowledge_gap(self, question):
        """Flag areas where knowledge is lacking"""
        if 'knowledge_gaps' not in self.knowledge_base:
            self.knowledge_base['knowledge_gaps'] = []
        self.knowledge_base['knowledge_gaps'].append({
            'question': question,
            'timestamp': datetime.now()
        })
        self.save_knowledge_base()
