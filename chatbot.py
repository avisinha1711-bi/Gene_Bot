"""
Enhanced Genetics Chatbot with advanced features for deployment.
This file provides additional functionality on top of the main Mistral-based bot.
"""

import requests
import json
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import hashlib

from config import chatbot_config
from knowledge_base import GeneticsKnowledgeBase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGeneticsChatBot:
    """
    Enhanced genetics chatbot with additional features:
    - PubMed integration for latest research
    - User feedback learning
    - Conversation analytics
    - FAQ management
    - Performance monitoring
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """Initialize enhanced genetics chatbot"""
        self.knowledge_base_path = knowledge_base_path or chatbot_config.knowledge_base_path
        
        # Initialize components
        self.knowledge_base = self._initialize_knowledge_base()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feedback_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        # Data structures
        self.user_feedback = []
        self.conversation_history = []
        self.faq_cache = {}
        self.research_cache = {}
        self.performance_metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'avg_response_time': 0,
            'feedback_stats': {'positive': 0, 'negative': 0, 'neutral': 0}
        }
        
        # Initialize vectorizers
        self._initialize_vectorizers()
        
        # Create directories
        self._ensure_directories()
        
        logger.info("Enhanced Genetics ChatBot initialized")
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        directories = [
            './data/feedback',
            './data/analytics',
            './data/cache',
            './data/models'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize or load knowledge base"""
        kb_path = Path(self.knowledge_base_path)
        
        try:
            # Try to load existing knowledge base
            pickle_path = kb_path / 'genetics_kb_enhanced.pkl'
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    kb = pickle.load(f)
                logger.info(f"Loaded existing knowledge base from {pickle_path}")
                return kb
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")
        
        # Initialize new knowledge base
        kb = {
            'genes': {},
            'diseases': {},
            'research_papers': {},
            'faq': {},
            'knowledge_gaps': [],
            'user_queries': [],
            'entity_relations': {},
            'temporal_data': {},
            'metadata': {
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        # Try to load from JSON if exists
        json_path = kb_path / 'genetics_kb.json'
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                kb.update(json_data)
                logger.info(f"Loaded data from {json_path}")
            except Exception as e:
                logger.warning(f"Could not load JSON knowledge base: {e}")
        
        return kb
    
    def _initialize_vectorizers(self):
        """Initialize TF-IDF vectorizers with existing knowledge"""
        try:
            # Extract text for vectorization
            texts = []
            
            # Add FAQ questions
            for question, answer in self.knowledge_base.get('faq', {}).items():
                texts.append(f"{question} {answer}")
            
            # Add gene descriptions
            for gene_id, gene_info in self.knowledge_base.get('genes', {}).items():
                if isinstance(gene_info, dict):
                    texts.append(gene_info.get('description', ''))
            
            # Add disease descriptions
            for disease_id, disease_info in self.knowledge_base.get('diseases', {}).items():
                if isinstance(disease_info, dict):
                    texts.append(disease_info.get('description', ''))
            
            if texts:
                self.vectorizer.fit(texts)
                logger.info(f"TF-IDF vectorizer fitted with {len(texts)} documents")
        
        except Exception as e:
            logger.error(f"Error initializing vectorizer: {e}")
    
    def save_knowledge_base(self):
        """Save knowledge base to disk"""
        try:
            # Update metadata
            self.knowledge_base['metadata']['last_updated'] = datetime.now().isoformat()
            self.knowledge_base['metadata']['version'] = '1.0.0'
            
            # Save to pickle
            pickle_path = Path(self.knowledge_base_path) / 'genetics_kb_enhanced.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.knowledge_base, f)
            
            # Save to JSON for readability
            json_path = Path(self.knowledge_base_path) / 'genetics_kb.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Knowledge base saved to {pickle_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def fetch_pubmed_data(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch latest genetics research from PubMed API"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.research_cache:
            if (datetime.now() - self.research_cache[cache_key]['timestamp']).hours < 24:
                logger.info(f"Returning cached PubMed results for: {query}")
                return self.research_cache[cache_key]['results']
        
        try:
            # Step 1: Search for articles
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': f"{query} AND (genetics OR genomics OR molecular biology)",
                'retmode': 'json',
                'retmax': max_results,
                'sort': 'relevance',
                'field': 'title,abstract'
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            search_data = response.json()
            
            article_ids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not article_ids:
                logger.info(f"No PubMed results found for: {query}")
                return []
            
            # Step 2: Fetch article details
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(article_ids),
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
            soup = BeautifulSoup(fetch_response.content, 'xml')
            
            # Parse results
            results = []
            for article in soup.find_all('PubmedArticle'):
                try:
                    # Extract article information
                    title_elem = article.find('ArticleTitle')
                    abstract_elem = article.find('AbstractText')
                    authors = article.find_all('Author')
                    pub_date = article.find('PubDate')
                    
                    if title_elem and abstract_elem:
                        article_data = {
                            'title': title_elem.text.strip(),
                            'abstract': abstract_elem.text.strip()[:500] + '...',
                            'authors': [f"{author.find('LastName').text if author.find('LastName') else ''} "
                                       f"{author.find('Initials').text if author.find('Initials') else ''}"
                                       for author in authors[:3]],  # First 3 authors
                            'pub_date': pub_date.text if pub_date else 'Unknown',
                            'pmid': article.find('PMID').text if article.find('PMID') else '',
                            'query_relevance': self._calculate_relevance(
                                query, 
                                f"{title_elem.text} {abstract_elem.text}"
                            )
                        }
                        
                        # Extract keywords if available
                        keywords = article.find_all('Keyword')
                        if keywords:
                            article_data['keywords'] = [k.text for k in keywords[:5]]
                        
                        results.append(article_data)
                
                except Exception as e:
                    logger.warning(f"Error parsing PubMed article: {e}")
                    continue
            
            # Sort by relevance
            results.sort(key=lambda x: x['query_relevance'], reverse=True)
            
            # Cache results
            self.research_cache[cache_key] = {
                'results': results[:max_results],
                'timestamp': datetime.now(),
                'query': query
            }
            
            logger.info(f"Fetched {len(results)} PubMed articles for: {query}")
            return results[:max_results]
        
        except requests.exceptions.Timeout:
            logger.error(f"PubMed API timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Error fetching PubMed data: {e}")
            return []
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text"""
        try:
            # Simple cosine similarity using TF-IDF
            query_vec = self.vectorizer.transform([query])
            text_vec = self.vectorizer.transform([text])
            
            if query_vec.shape[1] > 0 and text_vec.shape[1] > 0:
                similarity = cosine_similarity(query_vec, text_vec)[0][0]
                return float(similarity)
            
            # Fallback: keyword matching
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            intersection = query_words.intersection(text_words)
            
            return len(intersection) / len(query_words) if query_words else 0
        
        except Exception:
            return 0.0
    
    def extract_genetic_entities(self, text: str) -> Dict[str, List]:
        """Extract genetic entities from text"""
        entities = {
            'genes': self._extract_genes(text),
            'diseases': self._extract_diseases(text),
            'variants': self._extract_variants(text),
            'proteins': self._extract_proteins(text),
            'pathways': self._extract_pathways(text)
        }
        
        # Update knowledge base with new entities
        self._update_entity_knowledge(entities, text)
        
        return entities
    
    def _extract_genes(self, text: str) -> List[str]:
        """Extract gene symbols and names"""
        # Common gene patterns
        patterns = [
            r'\b[A-Z][A-Z0-9]+\b',  # Gene symbols (BRCA1, TP53)
            r'\b[A-Z]{1,2}\d{5,6}\b',  # Gene IDs
            r'gene\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Gene names
        ]
        
        genes = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if 2 <= len(match) <= 10 and match not in genes:
                    genes.append(match)
        
        return list(set(genes))
    
    def _extract_diseases(self, text: str) -> List[str]:
        """Extract disease names"""
        # Common disease patterns
        patterns = [
            r'\b[A-Z][a-z]+(?:\'s)?(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|disorder)\b',
            r'\b(?:Huntington\'s|Parkinson\'s|Alzheimer\'s)\b',
            r'\b(?:cancer|leukemia|lymphoma|melanoma)\b',
            r'\b[A-Z][a-z]+\s+cancer\b'
        ]
        
        diseases = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            diseases.extend(matches)
        
        return list(set(diseases))
    
    def _extract_variants(self, text: str) -> Dict[str, List]:
        """Extract genetic variants"""
        variants = {
            'rs_ids': re.findall(r'\brs\d+\b', text, re.IGNORECASE),
            'chromosomal_positions': re.findall(r'\bchr\d+[:\-]\d+(?:[:\-]\d+)?\b', text, re.IGNORECASE),
            'mutations': re.findall(r'\b[cC]\.[0-9]+[ACGT]>[ACGT]\b', text),
            'indels': re.findall(r'\b[cC]\.[0-9]+_[0-9]+(?:del|ins|dup)[ACGT]+\b', text)
        }
        
        return variants
    
    def _extract_proteins(self, text: str) -> List[str]:
        """Extract protein names"""
        patterns = [
            r'\b[A-Z][a-z]+(?:ase|in|ogen|phin)\b',  # Common protein suffixes
            r'\b(?:p53|EGFR|HER2|VEGF)\b',  # Common protein symbols
        ]
        
        proteins = []
        for pattern in patterns:
            proteins.extend(re.findall(pattern, text))
        
        return list(set(proteins))
    
    def _extract_pathways(self, text: str) -> List[str]:
        """Extract biological pathway names"""
        pathways = [
            'Wnt signaling pathway',
            'MAPK pathway',
            'PI3K/AKT pathway',
            'Notch signaling pathway',
            'Hedgehog signaling pathway',
            'TGF-beta pathway',
            'NF-kB pathway',
            'JAK-STAT pathway'
        ]
        
        found_pathways = []
        text_lower = text.lower()
        for pathway in pathways:
            if pathway.lower() in text_lower:
                found_pathways.append(pathway)
        
        return found_pathways
    
    def _update_entity_knowledge(self, entities: Dict, context: str):
        """Update knowledge base with extracted entities"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Update genes
            for gene in entities['genes']:
                if gene not in self.knowledge_base['genes']:
                    self.knowledge_base['genes'][gene] = {
                        'symbol': gene,
                        'first_seen': timestamp,
                        'contexts': [context[:200]],
                        'occurrence_count': 1
                    }
                else:
                    self.knowledge_base['genes'][gene]['occurrence_count'] += 1
                    self.knowledge_base['genes'][gene]['contexts'].append(context[:200])
                    if len(self.knowledge_base['genes'][gene]['contexts']) > 10:
                        self.knowledge_base['genes'][gene]['contexts'] = \
                            self.knowledge_base['genes'][gene]['contexts'][-10:]
            
            # Update diseases
            for disease in entities['diseases']:
                if disease not in self.knowledge_base['diseases']:
                    self.knowledge_base['diseases'][disease] = {
                        'name': disease,
                        'first_seen': timestamp,
                        'contexts': [context[:200]],
                        'occurrence_count': 1
                    }
            
            # Update entity relations
            for gene in entities['genes']:
                for disease in entities['diseases']:
                    relation_key = f"{gene}_{disease}"
                    if relation_key not in self.knowledge_base['entity_relations']:
                        self.knowledge_base['entity_relations'][relation_key] = {
                            'gene': gene,
                            'disease': disease,
                            'first_seen': timestamp,
                            'context': context[:200],
                            'occurrence_count': 1
                        }
                    else:
                        self.knowledge_base['entity_relations'][relation_key]['occurrence_count'] += 1
            
            # Auto-save periodically
            if self.performance_metrics['total_queries'] % 10 == 0:
                self.save_knowledge_base()
        
        except Exception as e:
            logger.error(f"Error updating entity knowledge: {e}")
    
    def learn_from_interaction(self, question: str, answer: str, 
                              user_feedback: str, additional_data: Optional[Dict] = None):
        """Learn from user interactions and feedback"""
        start_time = datetime.now()
        
        try:
            # Create learning record
            learning_data = {
                'question': question,
                'answer': answer,
                'user_feedback': user_feedback,
                'timestamp': datetime.now().isoformat(),
                'entities': self.extract_genetic_entities(question + ' ' + answer),
                'response_quality': self._assess_response_quality(answer),
                'additional_data': additional_data or {}
            }
            
            # Add to feedback history
            self.user_feedback.append(learning_data)
            
            # Update performance metrics
            self.performance_metrics['total_queries'] += 1
            self.performance_metrics['feedback_stats'][user_feedback] = \
                self.performance_metrics['feedback_stats'].get(user_feedback, 0) + 1
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['avg_response_time'] = (
                (self.performance_metrics['avg_response_time'] * 
                 (self.performance_metrics['total_queries'] - 1) + response_time) /
                self.performance_metrics['total_queries']
            )
            
            # Update knowledge base based on feedback
            if user_feedback.lower() == "positive":
                self._reinforce_knowledge(question, answer, learning_data['entities'])
                self.performance_metrics['successful_responses'] += 1
            elif user_feedback.lower() == "negative":
                self._flag_knowledge_gap(question, answer)
            
            # Update FAQ if similar questions appear frequently
            self._update_faq_system(question, answer)
            
            # Save analytics
            self._save_interaction_analytics(learning_data)
            
            # Retrain vectorizers periodically
            if len(self.user_feedback) % 50 == 0:
                self._retrain_vectorizers()
            
            logger.info(f"Learned from interaction: {question[:50]}...")
            return True
        
        except Exception as e:
            logger.error(f"Error in learn_from_interaction: {e}")
            return False
    
    def _assess_response_quality(self, answer: str) -> Dict[str, Any]:
        """Assess the quality of a response"""
        quality_metrics = {
            'length_score': min(len(answer.split()) / 100, 1.0),
            'complexity_score': self._calculate_text_complexity(answer),
            'clarity_score': self._calculate_clarity_score(answer),
            'completeness_score': 0.0,  # Would need reference answer
            'overall_score': 0.0
        }
        
        # Calculate overall score (weighted average)
        weights = {'length_score': 0.2, 'complexity_score': 0.3, 
                  'clarity_score': 0.5}
        quality_metrics['overall_score'] = sum(
            quality_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return quality_metrics
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        try:
            # Simple metrics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            
            if not words or not sentences:
                return 0.5
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Normalize scores
            sentence_score = min(avg_sentence_length / 20, 1.0)
            word_score = min(avg_word_length / 8, 1.0)
            
            return (sentence_score + word_score) / 2
        
        except Exception:
            return 0.5
    
    def _calculate_clarity_score(self, text: str) -> float:
        """Calculate clarity score of text"""
        try:
            # Simple clarity heuristics
            clarity_indicators = [
                (r'\bhowever\b|\bbut\b|\balthough\b', 0.1),  # Conjunctions
                (r'\btherefore\b|\bthus\b|\bconsequently\b', 0.1),  # Conclusions
                (r'\bfor example\b|\bsuch as\b|\bincluding\b', 0.15),  # Examples
                (r'\bin summary\b|\bin conclusion\b', 0.2),  # Summaries
                (r'\bimportant\b|\bkey\b|\bcritical\b', 0.1),  # Emphasis
            ]
            
            score = 0.5  # Base score
            
            for pattern, weight in clarity_indicators:
                if re.search(pattern, text, re.IGNORECASE):
                    score += weight
            
            return min(score, 1.0)
        
        except Exception:
            return 0.5
    
    def _reinforce_knowledge(self, question: str, answer: str, entities: Dict):
        """Reinforce correct knowledge patterns"""
        try:
            # Add to FAQ if it's a good question-answer pair
            question_hash = hashlib.md5(question.lower().encode()).hexdigest()
            
            if question_hash not in self.faq_cache:
                self.faq_cache[question_hash] = {
                    'question': question,
                    'answer': answer,
                    'entities': entities,
                    'first_seen': datetime.now().isoformat(),
                    'reinforcement_count': 1,
                    'last_reinforced': datetime.now().isoformat()
                }
            else:
                self.faq_cache[question_hash]['reinforcement_count'] += 1
                self.faq_cache[question_hash]['last_reinforced'] = datetime.now().isoformat()
                self.faq_cache[question_hash]['answer'] = answer  # Update with latest answer
            
            # Add to knowledge base FAQ if reinforced multiple times
            if self.faq_cache[question_hash]['reinforcement_count'] >= 3:
                self.knowledge_base['faq'][question] = {
                    'answer': answer,
                    'entities': entities,
                    'reinforcement_count': self.faq_cache[question_hash]['reinforcement_count'],
                    'first_added': datetime.now().isoformat()
                }
                logger.info(f"Added to FAQ: {question[:50]}...")
            
            # Update knowledge base with successful interaction
            self.knowledge_base['user_queries'].append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'feedback': 'positive',
                'entities': entities
            })
            
            # Limit user queries storage
            if len(self.knowledge_base['user_queries']) > 1000:
                self.knowledge_base['user_queries'] = \
                    self.knowledge_base['user_queries'][-1000:]
            
            # Save knowledge base
            self.save_knowledge_base()
        
        except Exception as e:
            logger.error(f"Error reinforcing knowledge: {e}")
    
    def _flag_knowledge_gap(self, question: str, answer: str):
        """Flag areas where knowledge is lacking"""
        try:
            gap_record = {
                'question': question,
                'provided_answer': answer,
                'timestamp': datetime.now().isoformat(),
                'entities': self.extract_genetic_entities(question),
                'user_feedback': 'negative',
                'priority': self._calculate_gap_priority(question)
            }
            
            # Add to knowledge gaps
            self.knowledge_base['knowledge_gaps'].append(gap_record)
            
            # Limit knowledge gaps storage
            if len(self.knowledge_base['knowledge_gaps']) > 100:
                # Keep highest priority gaps
                self.knowledge_base['knowledge_gaps'].sort(
                    key=lambda x: x.get('priority', 0), 
                    reverse=True
                )
                self.knowledge_base['knowledge_gaps'] = \
                    self.knowledge_base['knowledge_gaps'][:100]
            
            logger.info(f"Flagged knowledge gap: {question[:50]}...")
            
            # Save knowledge base
            self.save_knowledge_base()
        
        except Exception as e:
            logger.error(f"Error flagging knowledge gap: {e}")
    
    def _calculate_gap_priority(self, question: str) -> float:
        """Calculate priority score for knowledge gap"""
        try:
            priority_score = 0.0
            
            # Factor 1: Question length (longer questions might be more specific)
            words = question.split()
            priority_score += min(len(words) / 50, 1.0) * 0.3
            
            # Factor 2: Contains important genetic terms
            important_terms = ['cancer', 'therapy', 'treatment', 'diagnosis', 
                              'mutation', 'inheritance', 'prevention']
            term_count = sum(1 for term in important_terms if term.lower() in question.lower())
            priority_score += (term_count / len(important_terms)) * 0.4
            
            # Factor 3: Question frequency in knowledge base
            similar_questions = sum(
                1 for q in self.knowledge_base.get('user_queries', []) 
                if self._questions_are_similar(q.get('question', ''), question)
            )
            priority_score += min(similar_questions / 10, 1.0) * 0.3
            
            return priority_score
        
        except Exception:
            return 0.5
    
    def _questions_are_similar(self, q1: str, q2: str, threshold: float = 0.7) -> bool:
        """Check if two questions are similar using cosine similarity"""
        try:
            if not q1 or not q2:
                return False
            
            # Vectorize questions
            vectors = self.vectorizer.transform([q1, q2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return similarity >= threshold
        
        except Exception:
            # Fallback: simple word overlap
            words1 = set(q1.lower().split())
            words2 = set(q2.lower().split())
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            
            return (overlap / total) >= threshold if total > 0 else False
    
    def _update_faq_system(self, question: str, answer: str):
        """Update FAQ system with new question-answer pairs"""
        try:
            # Check if this question is already similar to existing FAQs
            for faq_question, faq_data in list(self.knowledge_base.get('faq', {}).items()):
                if self._questions_are_similar(faq_question, question):
                    # Update existing FAQ with new answer if better
                    current_quality = self._assess_response_quality(faq_data.get('answer', ''))
                    new_quality = self._assess_response_quality(answer)
                    
                    if new_quality['overall_score'] > current_quality['overall_score']:
                        self.knowledge_base['faq'][faq_question]['answer'] = answer
                        self.knowledge_base['faq'][faq_question]['last_updated'] = \
                            datetime.now().isoformat()
                        logger.info(f"Updated FAQ answer for: {faq_question[:50]}...")
                    
                    return
            
            # If not similar to existing FAQs, check if it should be added
            question_quality = self._assess_response_quality(question + ' ' + answer)
            
            if (question_quality['overall_score'] > 0.7 and 
                len(answer.split()) > 20):  # Reasonable answer length
                
                # Add to FAQ
                self.knowledge_base['faq'][question] = {
                    'answer': answer,
                    'quality_score': question_quality['overall_score'],
                    'added_date': datetime.now().isoformat(),
                    'entities': self.extract_genetic_entities(question + ' ' + answer)
                }
                
                logger.info(f"Added new FAQ: {question[:50]}...")
                
                # Save knowledge base
                self.save_knowledge_base()
        
        except Exception as e:
            logger.error(f"Error updating FAQ system: {e}")
    
    def _save_interaction_analytics(self, interaction_data: Dict):
        """Save interaction analytics to file"""
        try:
            analytics_path = Path('./data/analytics') / f"interaction_{datetime.now().strftime('%Y%m')}.json"
            
            # Load existing analytics
            analytics = []
            if analytics_path.exists():
                with open(analytics_path, 'r', encoding='utf-8') as f:
                    analytics = json.load(f)
            
            # Add new interaction
            analytics.append(interaction_data)
            
            # Save back to file
            with open(analytics_path, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved interaction analytics to {analytics_path}")
        
        except Exception as e:
            logger.error(f"Error saving interaction analytics: {e}")
    
    def _retrain_vectorizers(self):
        """Retrain TF-IDF vectorizers with updated data"""
        try:
            # Collect all text data
            texts = []
            
            # Add FAQ data
            for question, data in self.knowledge_base.get('faq', {}).items():
                texts.append(f"{question} {data.get('answer', '')}")
            
            # Add user queries
            for query in self.knowledge_base.get('user_queries', []):
                texts.append(f"{query.get('question', '')} {query.get('answer', '')}")
            
            # Add feedback data
            for feedback in self.user_feedback[-100:]:  # Last 100 feedbacks
                texts.append(f"{feedback.get('question', '')} {feedback.get('answer', '')}")
            
            if texts:
                self.vectorizer.fit(texts)
                logger.info(f"Retrained vectorizer with {len(texts)} documents")
        
        except Exception as e:
            logger.error(f"Error retraining vectorizers: {e}")
    
    def get_similar_questions(self, question: str, k: int = 5) -> List[Dict]:
        """Get similar questions from history"""
        try:
            # Collect all questions
            all_questions = []
            
            # Add FAQ questions
            for faq_q, faq_data in self.knowledge_base.get('faq', {}).items():
                all_questions.append({
                    'question': faq_q,
                    'answer': faq_data.get('answer', ''),
                    'source': 'faq',
                    'quality': faq_data.get('quality_score', 0.5)
                })
            
            # Add recent user queries
            for query in self.knowledge_base.get('user_queries', [])[-100:]:
                all_questions.append({
                    'question': query.get('question', ''),
                    'answer': query.get('answer', ''),
                    'source': 'user_query',
                    'quality': self._assess_response_quality(
                        query.get('question', '') + ' ' + query.get('answer', '')
                    )['overall_score']
                })
            
            if not all_questions:
                return []
            
            # Calculate similarities
            similarities = []
            query_vec = self.vectorizer.transform([question])
            
            for i, q_data in enumerate(all_questions):
                q_vec = self.vectorizer.transform([q_data['question']])
                similarity = cosine_similarity(query_vec, q_vec)[0][0]
                
                # Adjust by quality score
                adjusted_similarity = similarity * (0.5 + 0.5 * q_data['quality'])
                
                similarities.append((adjusted_similarity, i))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            # Return top k
            results = []
            for similarity, idx in similarities[:k]:
                if similarity > 0.3:  # Threshold
                    results.append({
                        **all_questions[idx],
                        'similarity': float(similarity)
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error getting similar questions: {e}")
            return []
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        report = {
            'metrics': self.performance_metrics.copy(),
            'knowledge_base_stats': {
                'faq_count': len(self.knowledge_base.get('faq', {})),
                'gene_count': len(self.knowledge_base.get('genes', {})),
                'disease_count': len(self.knowledge_base.get('diseases', {})),
                'knowledge_gaps': len(self.knowledge_base.get('knowledge_gaps', [])),
                'user_queries': len(self.knowledge_base.get('user_queries', []))
            },
            'cache_stats': {
                'faq_cache_size': len(self.faq_cache),
                'research_cache_size': len(self.research_cache),
                'feedback_count': len(self.user_feedback)
            },
            'timestamps': {
                'startup_time': self.knowledge_base.get('metadata', {}).get('created', ''),
                'last_updated': self.knowledge_base.get('metadata', {}).get('last_updated', ''),
                'current_time': datetime.now().isoformat()
            }
        }
        
        # Calculate additional metrics
        if self.performance_metrics['total_queries'] > 0:
            report['metrics']['success_rate'] = (
                self.performance_metrics['successful_responses'] / 
                self.performance_metrics['total_queries']
            )
        
        return report
    
    def export_data(self, export_path: str) -> bool:
        """Export all chatbot data"""
        try:
            export_data = {
                'knowledge_base': self.knowledge_base,
                'performance_metrics': self.performance_metrics,
                'faq_cache': self.faq_cache,
                'user_feedback': self.user_feedback[-1000:],  # Last 1000 feedbacks
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '1.0.0'
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported data to {export_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def clear_cache(self, cache_type: str = 'all') -> bool:
        """Clear specified cache"""
        try:
            if cache_type in ['all', 'research']:
                self.research_cache.clear()
                logger.info("Cleared research cache")
            
            if cache_type in ['all', 'faq']:
                self.faq_cache.clear()
                logger.info("Cleared FAQ cache")
            
            if cache_type in ['all', 'vectorizer']:
                self._initialize_vectorizers()
                logger.info("Reinitialized vectorizers")
            
            return True
        
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


# Utility functions
def create_sample_knowledge_base(output_path: str = './data/knowledge_base'):
    """Create a sample knowledge base for testing"""
    os.makedirs(output_path, exist_ok=True)
    
    sample_data = {
        'genes': {
            'BRCA1': {
                'symbol': 'BRCA1',
                'name': 'Breast Cancer type 1 susceptibility protein',
                'description': 'Tumor suppressor protein involved in DNA repair',
                'chromosome': '17',
                'function': 'DNA damage repair, cell cycle control',
                'associated_diseases': ['Breast cancer', 'Ovarian cancer']
            },
            'TP53': {
                'symbol': 'TP53',
                'name': 'Tumor protein p53',
                'description': 'Guardian of the genome, regulates cell division',
                'chromosome': '17',
                'function': 'Apoptosis, DNA repair, cell cycle arrest',
                'associated_diseases': ['Li-Fraumeni syndrome', 'Multiple cancers']
            }
        },
        'diseases': {
            'Cystic Fibrosis': {
                'name': 'Cystic Fibrosis',
                'description': 'Genetic disorder affecting lungs and digestive system',
                'inheritance': 'Autosomal recessive',
                'caused_by': 'Mutations in CFTR gene',
                'symptoms': ['Lung infections', 'Digestive problems', 'Poor growth']
            }
        },
        'faq': {
            'What is DNA?': {
                'answer': 'DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for development, functioning, growth and reproduction of all known organisms.',
                'quality_score': 0.9,
                'added_date': datetime.now().isoformat()
            }
        },
        'metadata': {
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    }
    
    # Save sample data
    with open(os.path.join(output_path, 'genetics_kb.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample knowledge base created at {output_path}")
    return sample_data


def main():
    """Test the enhanced genetics chatbot"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced Genetics ChatBot')
    parser.add_argument('--create-sample', action='store_true', help='Create sample knowledge base')
    parser.add_argument('--test-pubmed', type=str, help='Test PubMed API with query')
    parser.add_argument('--test-entities', type=str, help='Test entity extraction')
    parser.add_argument('--performance', action='store_true', help='Show performance report')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_knowledge_base()
        return
    
    # Initialize chatbot
    chatbot = EnhancedGeneticsChatBot()
    
    if args.test_pubmed:
        print(f"\nTesting PubMed API for: {args.test_pubmed}")
        results = chatbot.fetch_pubmed_data(args.test_pubmed, max_results=3)
        
        if results:
            for i, article in enumerate(results, 1):
                print(f"\n{i}. {article['title']}")
                print(f"   Abstract: {article['abstract']}")
                print(f"   Authors: {', '.join(article['authors'])}")
                print(f"   Relevance: {article['query_relevance']:.3f}")
        else:
            print("No results found")
    
    elif args.test_entities:
        print(f"\nTesting entity extraction for: {args.test_entities}")
        entities = chatbot.extract_genetic_entities(args.test_entities)
        
        print("\nExtracted Entities:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                if isinstance(entity_list, list):
                    print(f"  {entity_type}: {entity_list}")
                elif isinstance(entity_list, dict):
                    print(f"  {entity_type}:")
                    for sub_type, sub_list in entity_list.items():
                        if sub_list:
                            print(f"    {sub_type}: {sub_list}")
    
    elif args.performance:
        report = chatbot.get_performance_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2))
    
    else:
        # Interactive mode
        print("Enhanced Genetics ChatBot - Test Interface")
        print("Commands: quit, stats, clear, export, help")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                elif user_input.lower() == 'stats':
                    report = chatbot.get_performance_report()
                    print(f"\nTotal queries: {report['metrics']['total_queries']}")
                    print(f"FAQ count: {report['knowledge_base_stats']['faq_count']}")
                    print(f"Success rate: {report['metrics'].get('success_rate', 0):.2%}")
                elif user_input.lower() == 'clear':
                    chatbot.clear_cache('all')
                    print("Cache cleared")
                elif user_input.lower() == 'export':
                    chatbot.export_data('./data/export.json')
                    print("Data exported to ./data/export.json")
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  <question> - Ask a genetics question")
                    print("  stats - Show performance statistics")
                    print("  clear - Clear cache")
                    print("  export - Export all data")
                    print("  quit - Exit the program")
                elif user_input:
                    # Test entity extraction
                    entities = chatbot.extract_genetic_entities(user_input)
                    
                    # Test PubMed lookup for genetic terms
                    if entities['genes']:
                        gene = entities['genes'][0]
                        print(f"\nFound gene: {gene}")
                        papers = chatbot.fetch_pubmed_data(gene, max_results=2)
                        if papers:
                            print(f"Recent research on {gene}:")
                            for paper in papers:
                                print(f"  • {paper['title'][:80]}...")
                    
                    # Get similar questions
                    similar = chatbot.get_similar_questions(user_input)
                    if similar:
                        print(f"\nSimilar questions in database:")
                        for i, sim_q in enumerate(similar[:3], 1):
                            print(f"  {i}. {sim_q['question'][:60]}...")
                    
                    # Simulate learning
                    feedback = input("\nWas this helpful? (positive/negative/neutral): ").strip().lower()
                    if feedback in ['positive', 'negative', 'neutral']:
                        chatbot.learn_from_interaction(
                            user_input,
                            "Sample response based on available knowledge",
                            feedback
                        )
                        print(f"Feedback recorded: {feedback}")
                    else:
                        print("Invalid feedback")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
