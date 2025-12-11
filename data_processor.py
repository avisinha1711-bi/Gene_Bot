"""
Advanced Genetics Data Processor for deployment.
Handles data extraction, processing, and integration with various genetics databases.
"""

import re
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from pathlib import Path
import hashlib
import pickle
import gzip
import csv

from config import chatbot_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeneticsDataProcessor:
    """
    Advanced data processor for genetics information extraction and integration.
    Supports multiple databases: NCBI, Ensembl, UniProt, ClinVar, and more.
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize genetics data processor"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Compile regex patterns for entity extraction
        self._compile_patterns()
        
        # Database endpoints
        self.databases = {
            'ncbi_gene': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            'ensembl': 'https://rest.ensembl.org/',
            'uniprot': 'https://www.ebi.ac.uk/proteins/api/',
            'clinvar': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            'pubmed': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            'omim': 'https://api.omim.org/api/',
            'gnomad': 'https://gnomad.broadinstitute.org/api/',
            'cancer_gene_census': 'https://cancer.sanger.ac.uk/cosmic/'
        }
        
        # Cache configuration
        self.cache_enabled = True
        self.cache_expiry = timedelta(hours=24)
        self.request_timeout = 30
        self.max_retries = 3
        
        # Rate limiting
        self.request_delay = 0.1  # seconds between requests
        self.last_request_time = datetime.now()
        
        # Initialize session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GeneticsChatBot/1.0 (https://github.com/yourusername/genetics-chatbot)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        # Entity dictionaries for validation
        self._load_entity_dictionaries()
        
        logger.info("Genetics Data Processor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for entity extraction"""
        # Gene patterns
        self.gene_patterns = [
            re.compile(r'\b[A-Z][A-Z0-9]+\b'),  # Standard gene symbols (BRCA1, TP53)
            re.compile(r'\b[A-Z]{1,2}\d{5,6}\b'),  # NCBI gene IDs
            re.compile(r'\bgene\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'),  # Gene names
            re.compile(r'\bENSG\d{11}\b'),  # Ensembl gene IDs
            re.compile(r'\b[A-Z][a-z]+\d*\b'),  # Additional gene patterns
        ]
        
        # Disease patterns
        self.disease_patterns = [
            re.compile(r'\b[A-Z][a-z]+(?:\'s)?(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|disorder)\b', re.IGNORECASE),
            re.compile(r'\b(?:Huntington\'s|Parkinson\'s|Alzheimer\'s|Tay-Sachs|Marfan)\b', re.IGNORECASE),
            re.compile(r'\b(?:cancer|leukemia|lymphoma|melanoma|sarcoma|carcinoma)\b', re.IGNORECASE),
            re.compile(r'\b[A-Z][a-z]+\s+cancer\b', re.IGNORECASE),
            re.compile(r'\b(?:Type\s+I{1,3}|Type\s+\d+)\s+[A-Z][a-z]+\b', re.IGNORECASE),
        ]
        
        # Variant patterns
        self.variant_patterns = {
            'rs_ids': re.compile(r'\brs\d+\b', re.IGNORECASE),
            'chromosomal_positions': re.compile(r'\b(?:chr)?(?:[1-9]|1[0-9]|2[0-2]|X|Y|M|MT)[:\-]\d+(?:[:\-]\d+)?\b', re.IGNORECASE),
            'hgvs': re.compile(r'\b(?:c|g|n|p|r)\.\d+[ACGTUacgtu_]+(?:>|del|ins|dup|inv)[ACGTUacgtu_]*\b'),
            'cosmic': re.compile(r'\bCOSM\d+\b', re.IGNORECASE),
            'mutations': re.compile(r'\b(?:p\.)?[A-Z][a-z]{2}\d+[A-Z][a-z]{2}\b'),
        }
        
        # Protein patterns
        self.protein_patterns = [
            re.compile(r'\b[A-Z][a-z]+(?:ase|in|ogen|phin|lin|mycin)\b'),
            re.compile(r'\b(?:p53|EGFR|HER2|VEGF|CFTR|BRCA[12]|APOE)\b'),
            re.compile(r'\bENSP\d{11}\b'),  # Ensembl protein IDs
            re.compile(r'\b[A-Z0-9]{6,10}\b'),  # UniProt IDs
        ]
        
        # Chemical/drug patterns
        self.chemical_patterns = [
            re.compile(r'\b[A-Z][a-z]+\b(?:[-\s][A-Z]?[a-z]+)*\b(?:ib|ab|mab|tinib|zumab)\b'),  # Drug names
            re.compile(r'\b\d+-\d+-\d+\b'),  # CAS numbers
        ]
        
        # Pathway patterns
        self.pathway_patterns = [
            re.compile(r'\b(?:Wnt|MAPK|PI3K/AKT|Notch|Hedgehog|TGF-beta|NF-?kB|JAK-STAT)\s+(?:signaling\s+)?pathway\b', re.IGNORECASE),
            re.compile(r'\bKEGG\s+\d{5}\b'),
            re.compile(r'\bREACTOME\s+R-\w+-\d+\b'),
        ]
    
    def _load_entity_dictionaries(self):
        """Load entity dictionaries for validation"""
        self.entity_dictionaries = {
            'genes': set(),
            'diseases': set(),
            'proteins': set(),
            'chemicals': set(),
        }
        
        # Try to load from cache
        cache_file = self.cache_dir / 'entity_dicts.pkl.gz'
        if cache_file.exists():
            try:
                with gzip.open(cache_file, 'rb') as f:
                    self.entity_dictionaries = pickle.load(f)
                logger.info(f"Loaded entity dictionaries from cache ({len(self.entity_dictionaries['genes'])} genes)")
            except Exception as e:
                logger.warning(f"Could not load entity dictionaries: {e}")
    
    def _save_entity_dictionaries(self):
        """Save entity dictionaries to cache"""
        try:
            cache_file = self.cache_dir / 'entity_dicts.pkl.gz'
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(self.entity_dictionaries, f)
            logger.info("Saved entity dictionaries to cache")
        except Exception as e:
            logger.error(f"Error saving entity dictionaries: {e}")
    
    def extract_genetic_entities(self, text: str, validate: bool = True) -> Dict[str, Any]:
        """
        Extract all genetic entities from text with validation.
        
        Args:
            text: Input text to process
            validate: Whether to validate extracted entities against databases
        
        Returns:
            Dictionary containing extracted entities with metadata
        """
        start_time = datetime.now()
        
        try:
            # Basic entity extraction
            entities = {
                'genes': self._extract_genes(text),
                'diseases': self._extract_diseases(text),
                'variants': self._extract_variants(text),
                'proteins': self._extract_proteins(text),
                'chemicals': self._extract_chemicals(text),
                'pathways': self._extract_pathways(text),
                'metadata': {
                    'text_length': len(text),
                    'extraction_time': None,
                    'entity_counts': {},
                    'validation_status': 'pending'
                }
            }
            
            # Calculate entity counts
            for key, value in entities.items():
                if key != 'metadata':
                    if isinstance(value, dict):
                        entities['metadata']['entity_counts'][key] = sum(len(v) for v in value.values())
                    else:
                        entities['metadata']['entity_counts'][key] = len(value)
            
            # Validate entities if requested
            if validate and entities['metadata']['entity_counts']['genes'] > 0:
                entities = self._validate_entities(entities)
                entities['metadata']['validation_status'] = 'completed'
            
            # Calculate extraction time
            entities['metadata']['extraction_time'] = (datetime.now() - start_time).total_seconds()
            
            # Store in cache for future reference
            self._cache_entities(text, entities)
            
            logger.info(f"Extracted {sum(entities['metadata']['entity_counts'].values())} entities from text")
            return entities
        
        except Exception as e:
            logger.error(f"Error extracting genetic entities: {e}")
            return {
                'genes': [], 'diseases': [], 'variants': {}, 
                'proteins': [], 'chemicals': [], 'pathways': [],
                'metadata': {'error': str(e), 'extraction_time': 0, 'entity_counts': {}}
            }
    
    def _extract_genes(self, text: str) -> List[str]:
        """Extract gene symbols and names from text"""
        genes = set()
        
        for pattern in self.gene_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                # Filter out common false positives
                if self._is_valid_gene(match):
                    genes.add(match.upper())
        
        # Additional filtering based on known gene list
        if self.entity_dictionaries['genes']:
            genes = {g for g in genes if g in self.entity_dictionaries['genes']}
        
        return sorted(genes)
    
    def _is_valid_gene(self, candidate: str) -> bool:
        """Validate if a string is likely a gene symbol"""
        # Length constraints
        if len(candidate) < 2 or len(candidate) > 15:
            return False
        
        # Common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT',
            'DNA', 'RNA', 'PCR', 'FDA', 'CDC', 'WHO', 'USA', 'UK'
        }
        
        if candidate.upper() in false_positives:
            return False
        
        # Check pattern (usually starts with letter, contains letters/numbers)
        if not re.match(r'^[A-Z][A-Z0-9]*$', candidate.upper()):
            return False
        
        return True
    
    def _extract_diseases(self, text: str) -> List[str]:
        """Extract disease names from text"""
        diseases = set()
        
        for pattern in self.disease_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                diseases.add(match)
        
        return sorted(diseases)
    
    def _extract_variants(self, text: str) -> Dict[str, List]:
        """Extract genetic variants from text"""
        variants = {}
        
        for variant_type, pattern in self.variant_patterns.items():
            matches = pattern.findall(text)
            variants[variant_type] = list(set(matches))
        
        return variants
    
    def _extract_proteins(self, text: str) -> List[str]:
        """Extract protein names from text"""
        proteins = set()
        
        for pattern in self.protein_patterns:
            matches = pattern.findall(text)
            proteins.update(matches)
        
        return sorted(proteins)
    
    def _extract_chemicals(self, text: str) -> List[str]:
        """Extract chemical/drug names from text"""
        chemicals = set()
        
        for pattern in self.chemical_patterns:
            matches = pattern.findall(text)
            chemicals.update(matches)
        
        return sorted(chemicals)
    
    def _extract_pathways(self, text: str) -> List[str]:
        """Extract biological pathway names from text"""
        pathways = set()
        
        for pattern in self.pathway_patterns:
            matches = pattern.findall(text)
            pathways.update(matches)
        
        return sorted(pathways)
    
    def _validate_entities(self, entities: Dict) -> Dict:
        """Validate extracted entities against external databases"""
        validated_entities = entities.copy()
        
        try:
            # Validate genes
            if entities['genes']:
                validated_genes = []
                for gene in entities['genes']:
                    validation_result = self.validate_gene(gene)
                    if validation_result.get('valid', False):
                        validated_genes.append({
                            'symbol': gene,
                            'validation': validation_result
                        })
                validated_entities['genes'] = validated_genes
            
            # Validate diseases
            if entities['diseases']:
                validated_diseases = []
                for disease in entities['diseases']:
                    validation_result = self.validate_disease(disease)
                    if validation_result.get('valid', False):
                        validated_diseases.append({
                            'name': disease,
                            'validation': validation_result
                        })
                validated_entities['diseases'] = validated_diseases
            
            # Add validation metadata
            validated_entities['metadata']['validation_timestamp'] = datetime.now().isoformat()
            validated_entities['metadata']['validated_entities'] = {
                'genes': len(validated_entities.get('genes', [])),
                'diseases': len(validated_entities.get('diseases', []))
            }
            
        except Exception as e:
            logger.error(f"Error validating entities: {e}")
            validated_entities['metadata']['validation_error'] = str(e)
        
        return validated_entities
    
    @lru_cache(maxsize=1000)
    def validate_gene(self, gene_symbol: str) -> Dict[str, Any]:
        """
        Validate a gene symbol against multiple databases.
        
        Args:
            gene_symbol: Gene symbol to validate
        
        Returns:
            Dictionary with validation results
        """
        cache_key = f"gene_validation_{gene_symbol}"
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return cached_result
        
        validation_result = {
            'symbol': gene_symbol,
            'valid': False,
            'sources': [],
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Try multiple databases
            databases_to_check = [
                ('ncbi', self._validate_gene_ncbi),
                ('ensembl', self._validate_gene_ensembl),
                ('uniprot', self._validate_gene_uniprot)
            ]
            
            for db_name, validator in databases_to_check:
                try:
                    result = validator(gene_symbol)
                    if result.get('valid', False):
                        validation_result['valid'] = True
                        validation_result['sources'].append(db_name)
                        validation_result['details'][db_name] = result
                        
                        # Add to entity dictionary
                        self.entity_dictionaries['genes'].add(gene_symbol.upper())
                        
                        # Break if we have a valid result
                        if len(validation_result['sources']) >= 2:
                            break
                
                except Exception as e:
                    logger.debug(f"Error validating gene {gene_symbol} with {db_name}: {e}")
            
            # Cache the result
            self._save_to_cache(cache_key, validation_result)
            
            return validation_result
        
        except Exception as e:
            logger.error(f"Error in gene validation for {gene_symbol}: {e}")
            validation_result['error'] = str(e)
            return validation_result
    
    def _validate_gene_ncbi(self, gene_symbol: str) -> Dict[str, Any]:
        """Validate gene using NCBI Entrez API"""
        try:
            url = f"{self.databases['ncbi_gene']}esearch.fcgi"
            params = {
                'db': 'gene',
                'term': f"{gene_symbol}[Gene Name] AND human[Organism]",
                'retmode': 'json',
                'retmax': 1
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            count = int(data.get('esearchresult', {}).get('count', 0))
            
            if count > 0:
                gene_id = data['esearchresult']['idlist'][0]
                
                # Fetch additional details
                details = self._fetch_gene_details_ncbi(gene_id)
                
                return {
                    'valid': True,
                    'gene_id': gene_id,
                    'source': 'NCBI',
                    'count': count,
                    'details': details
                }
        
        except Exception as e:
            logger.debug(f"NCBI validation failed for {gene_symbol}: {e}")
        
        return {'valid': False, 'source': 'NCBI'}
    
    def _fetch_gene_details_ncbi(self, gene_id: str) -> Dict[str, Any]:
        """Fetch detailed gene information from NCBI"""
        cache_key = f"gene_details_ncbi_{gene_id}"
        cached = self._get_from_cache(cache_key)
        
        if cached:
            return cached
        
        try:
            url = f"{self.databases['ncbi_gene']}esummary.fcgi"
            params = {
                'db': 'gene',
                'id': gene_id,
                'retmode': 'json'
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            result = data.get('result', {}).get(gene_id, {})
            
            details = {
                'name': result.get('name', ''),
                'description': result.get('description', ''),
                'chromosome': result.get('chromosome', ''),
                'map_location': result.get('maplocation', ''),
                'summary': result.get('summary', ''),
                'aliases': result.get('otheraliases', '').split(', ') if result.get('otheraliases') else [],
                'taxid': result.get('taxid', '')
            }
            
            self._save_to_cache(cache_key, details)
            return details
        
        except Exception as e:
            logger.debug(f"Error fetching NCBI gene details: {e}")
            return {}
    
    def _validate_gene_ensembl(self, gene_symbol: str) -> Dict[str, Any]:
        """Validate gene using Ensembl REST API"""
        try:
            url = f"{self.databases['ensembl']}lookup/symbol/homo_sapiens/{gene_symbol}"
            params = {'expand': 1}
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=self.request_timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'valid': True,
                    'gene_id': data.get('id', ''),
                    'source': 'Ensembl',
                    'biotype': data.get('biotype', ''),
                    'description': data.get('description', ''),
                    'location': f"{data.get('seq_region_name', '')}:{data.get('start', '')}-{data.get('end', '')}"
                }
        
        except Exception as e:
            logger.debug(f"Ensembl validation failed for {gene_symbol}: {e}")
        
        return {'valid': False, 'source': 'Ensembl'}
    
    def _validate_gene_uniprot(self, gene_symbol: str) -> Dict[str, Any]:
        """Validate gene using UniProt API"""
        try:
            url = f"{self.databases['uniprot']}proteins"
            params = {
                'gene': gene_symbol,
                'organism': 'human',
                'format': 'json',
                'size': 1
            }
            
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    protein = data[0]
                    
                    return {
                        'valid': True,
                        'source': 'UniProt',
                        'uniprot_id': protein.get('accession', ''),
                        'protein_name': protein.get('protein', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                        'gene_name': protein.get('gene', {}).get('name', {}).get('value', '')
                    }
        
        except Exception as e:
            logger.debug(f"UniProt validation failed for {gene_symbol}: {e}")
        
        return {'valid': False, 'source': 'UniProt'}
    
    def validate_disease(self, disease_name: str) -> Dict[str, Any]:
        """
        Validate a disease name against medical databases.
        
        Args:
            disease_name: Disease name to validate
        
        Returns:
            Dictionary with validation results
        """
        cache_key = f"disease_validation_{disease_name.lower().replace(' ', '_')}"
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return cached_result
        
        validation_result = {
            'name': disease_name,
            'valid': False,
            'sources': [],
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check against known disease patterns
            if any(pattern.search(disease_name) for pattern in self.disease_patterns):
                validation_result['valid'] = True
                validation_result['sources'].append('pattern_matching')
                validation_result['details']['pattern_matching'] = {
                    'matched_patterns': [p.pattern for p in self.disease_patterns if p.search(disease_name)]
                }
            
            # Try to fetch from OMIM (if API key available)
            omim_result = self._validate_disease_omim(disease_name)
            if omim_result.get('valid', False):
                validation_result['valid'] = True
                validation_result['sources'].append('omim')
                validation_result['details']['omim'] = omim_result
            
            # Add to entity dictionary if valid
            if validation_result['valid']:
                self.entity_dictionaries['diseases'].add(disease_name)
            
            # Cache the result
            self._save_to_cache(cache_key, validation_result)
            
            return validation_result
        
        except Exception as e:
            logger.error(f"Error in disease validation for {disease_name}: {e}")
            validation_result['error'] = str(e)
            return validation_result
    
    def _validate_disease_omim(self, disease_name: str) -> Dict[str, Any]:
        """Validate disease using OMIM API (requires API key)"""
        # This is a placeholder - OMIM requires an API key
        # In production, you would implement proper OMIM API calls
        
        # For now, do a simple text match against common diseases
        common_diseases = {
            'cancer', 'leukemia', 'lymphoma', 'diabetes', 'alzheimer', 'parkinson',
            'huntington', 'cystic fibrosis', 'sickle cell', 'down syndrome',
            'turner syndrome', 'klinefelter syndrome', 'marfan syndrome'
        }
        
        disease_lower = disease_name.lower()
        
        for common in common_diseases:
            if common in disease_lower:
                return {
                    'valid': True,
                    'source': 'OMIM',
                    'matched_term': common,
                    'confidence': 'medium'
                }
        
        return {'valid': False, 'source': 'OMIM'}
    
    def fetch_gene_info(self, gene_symbol: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Fetch comprehensive gene information from multiple sources.
        
        Args:
            gene_symbol: Gene symbol to fetch information for
            detailed: Whether to fetch detailed information
        
        Returns:
            Dictionary with gene information from multiple sources
        """
        cache_key = f"gene_info_{gene_symbol}_{detailed}"
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return cached_result
        
        gene_info = {
            'symbol': gene_symbol,
            'sources': {},
            'summary': '',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Fetch from multiple sources in parallel (using asyncio for efficiency)
            sources = [
                ('ncbi', self._fetch_gene_info_ncbi),
                ('ensembl', self._fetch_gene_info_ensembl),
                ('uniprot', self._fetch_gene_info_uniprot)
            ]
            
            # For now, fetch sequentially (can be optimized with async)
            for source_name, fetcher in sources:
                try:
                    info = fetcher(gene_symbol, detailed)
                    if info:
                        gene_info['sources'][source_name] = info
                except Exception as e:
                    logger.debug(f"Error fetching from {source_name} for {gene_symbol}: {e}")
            
            # Generate summary
            gene_info['summary'] = self._generate_gene_summary(gene_info['sources'])
            
            # Cache the result
            self._save_to_cache(cache_key, gene_info)
            
            return gene_info
        
        except Exception as e:
            logger.error(f"Error fetching gene info for {gene_symbol}: {e}")
            gene_info['error'] = str(e)
            return gene_info
    
    def _fetch_gene_info_ncbi(self, gene_symbol: str, detailed: bool = False) -> Dict[str, Any]:
        """Fetch gene information from NCBI"""
        try:
            # First search for gene ID
            search_url = f"{self.databases['ncbi_gene']}esearch.fcgi"
            search_params = {
                'db': 'gene',
                'term': f"{gene_symbol}[Gene Name] AND human[Organism]",
                'retmode': 'json',
                'retmax': 1
            }
            
            search_response = self._make_request(search_url, search_params)
            search_data = search_response.json()
            
            if int(search_data.get('esearchresult', {}).get('count', 0)) == 0:
                return {}
            
            gene_id = search_data['esearchresult']['idlist'][0]
            
            # Fetch summary
            summary_url = f"{self.databases['ncbi_gene']}esummary.fcgi"
            summary_params = {
                'db': 'gene',
                'id': gene_id,
                'retmode': 'json'
            }
            
            summary_response = self._make_request(summary_url, summary_params)
            summary_data = summary_response.json()
            
            result = summary_data.get('result', {}).get(gene_id, {})
            
            info = {
                'gene_id': gene_id,
                'name': result.get('name', ''),
                'description': result.get('description', ''),
                'summary': result.get('summary', ''),
                'chromosome': result.get('chromosome', ''),
                'map_location': result.get('maplocation', ''),
                'aliases': result.get('otheraliases', '').split(', ') if result.get('otheraliases') else []
            }
            
            # Fetch detailed information if requested
            if detailed:
                # Fetch PubMed citations
                pubmed_url = f"{self.databases['pubmed']}esearch.fcgi"
                pubmed_params = {
                    'db': 'pubmed',
                    'term': f"{gene_symbol}[TIAB] AND human[MeSH]",
                    'retmode': 'json',
                    'retmax': 5,
                    'sort': 'relevance'
                }
                
                pubmed_response = self._make_request(pubmed_url, pubmed_params)
                pubmed_data = pubmed_response.json()
                
                pubmed_ids = pubmed_data.get('esearchresult', {}).get('idlist', [])
                info['pubmed_citations'] = len(pubmed_ids)
                info['recent_pubmed_ids'] = pubmed_ids[:3]
                
                # Fetch clinical variants from ClinVar
                clinvar_url = f"{self.databases['clinvar']}esearch.fcgi"
                clinvar_params = {
                    'db': 'clinvar',
                    'term': f"{gene_symbol}[Gene Name]",
                    'retmode': 'json',
                    'retmax': 1
                }
                
                clinvar_response = self._make_request(clinvar_url, clinvar_params)
                clinvar_data = clinvar_response.json()
                
                clinvar_count = int(clinvar_data.get('esearchresult', {}).get('count', 0))
                info['clinvar_variants'] = clinvar_count
            
            return info
        
        except Exception as e:
            logger.debug(f"Error fetching NCBI gene info for {gene_symbol}: {e}")
            return {}
    
    def _fetch_gene_info_ensembl(self, gene_symbol: str, detailed: bool = False) -> Dict[str, Any]:
        """Fetch gene information from Ensembl"""
        try:
            url = f"{self.databases['ensembl']}lookup/symbol/homo_sapiens/{gene_symbol}"
            params = {'expand': 1} if detailed else {}
            
            headers = {'Content-Type': 'application/json'}
            
            response = self.session.get(url, params=params, headers=headers, timeout=self.request_timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                info = {
                    'ensembl_id': data.get('id', ''),
                    'biotype': data.get('biotype', ''),
                    'description': data.get('description', ''),
                    'location': {
                        'chromosome': data.get('seq_region_name', ''),
                        'start': data.get('start', ''),
                        'end': data.get('end', ''),
                        'strand': data.get('strand', '')
                    },
                    'transcript_count': len(data.get('Transcript', [])) if detailed else 0
                }
                
                return info
        
        except Exception as e:
            logger.debug(f"Error fetching Ensembl gene info for {gene_symbol}: {e}")
        
        return {}
    
    def _fetch_gene_info_uniprot(self, gene_symbol: str, detailed: bool = False) -> Dict[str, Any]:
        """Fetch gene information from UniProt"""
        try:
            url = f"{self.databases['uniprot']}proteins"
            params = {
                'gene': gene_symbol,
                'organism': 'human',
                'format': 'json',
                'size': 1
            }
            
            if detailed:
                params['include'] = 'sequence,features'
            
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    protein = data[0]
                    
                    info = {
                        'uniprot_id': protein.get('accession', ''),
                        'protein_name': protein.get('protein', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                        'gene_name': protein.get('gene', {}).get('name', {}).get('value', ''),
                        'organism': protein.get('organism', {}).get('scientificName', ''),
                        'sequence_length': len(protein.get('sequence', {}).get('sequence', '')) if detailed else 0
                    }
                    
                    if detailed:
                        info['keywords'] = [kw.get('value', '') for kw in protein.get('keywords', [])]
                        info['features'] = [feat.get('description', '') for feat in protein.get('features', [])[:5]]
                    
                    return info
        
        except Exception as e:
            logger.debug(f"Error fetching UniProt gene info for {gene_symbol}: {e}")
        
        return {}
    
    def _generate_gene_summary(self, sources_info: Dict[str, Any]) -> str:
        """Generate a human-readable summary from multiple sources"""
        summaries = []
        
        if 'ncbi' in sources_info:
            ncbi = sources_info['ncbi']
            if ncbi.get('summary'):
                summaries.append(ncbi['summary'])
            elif ncbi.get('description'):
                summaries.append(ncbi['description'])
        
        if 'ensembl' in sources_info:
            ensembl = sources_info['ensembl']
            if ensembl.get('description'):
                summaries.append(ensembl['description'])
        
        if 'uniprot' in sources_info:
            uniprot = sources_info['uniprot']
            if uniprot.get('protein_name'):
                summaries.append(f"Encodes {uniprot['protein_name']}")
        
        if summaries:
            # Take the longest summary (usually most detailed)
            return max(summaries, key=len)
        
        return "Gene information not available from sources"
    
    def fetch_variant_info(self, variant_id: str) -> Dict[str, Any]:
        """
        Fetch variant information from databases.
        
        Args:
            variant_id: Variant ID (rs number, COSMIC ID, etc.)
        
        Returns:
            Dictionary with variant information
        """
        cache_key = f"variant_info_{variant_id}"
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return cached_result
        
        variant_info = {
            'id': variant_id,
            'type': self._identify_variant_type(variant_id),
            'sources': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if variant_info['type'] == 'rs_id':
                # Fetch from dbSNP
                variant_info['sources']['dbsnp'] = self._fetch_rs_variant_info(variant_id)
            
            elif variant_info['type'] == 'cosmic':
                # Fetch from COSMIC
                variant_info['sources']['cosmic'] = self._fetch_cosmic_variant_info(variant_id)
            
            elif variant_info['type'] == 'hgvs':
                # Fetch using HGVS notation
                variant_info['sources']['hgvs'] = self._fetch_hgvs_variant_info(variant_id)
            
            # Also check ClinVar for clinical significance
            clinvar_info = self._fetch_clinvar_variant_info(variant_id)
            if clinvar_info:
                variant_info['sources']['clinvar'] = clinvar_info
            
            # Cache the result
            self._save_to_cache(cache_key, variant_info)
            
            return variant_info
        
        except Exception as e:
            logger.error(f"Error fetching variant info for {variant_id}: {e}")
            variant_info['error'] = str(e)
            return variant_info
    
    def _identify_variant_type(self, variant_id: str) -> str:
        """Identify the type of variant ID"""
        variant_id_lower = variant_id.lower()
        
        if variant_id_lower.startswith('rs'):
            return 'rs_id'
        elif variant_id_lower.startswith('cosm'):
            return 'cosmic'
        elif ':' in variant_id or '.' in variant_id:
            return 'hgvs'
        elif variant_id_lower.startswith('chr'):
            return 'chromosomal_position'
        else:
            return 'unknown'
    
    def _fetch_rs_variant_info(self, rs_id: str) -> Dict[str, Any]:
        """Fetch variant information from dbSNP"""
        try:
            url = f"{self.databases['ncbi_gene']}esummary.fcgi"
            params = {
                'db': 'snp',
                'id': rs_id[2:],  # Remove 'rs' prefix
                'retmode': 'json'
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            result = data.get('result', {}).get(rs_id[2:], {})
            
            return {
                'chromosome': result.get('chr', ''),
                'position': result.get('chrpos', ''),
                'alleles': result.get('alleles', ''),
                'maf': result.get('global_maf', ''),
                'genes': result.get('genes', []),
                'clinical_significance': result.get('clinical_significance', '')
            }
        
        except Exception as e:
            logger.debug(f"Error fetching dbSNP info for {rs_id}: {e}")
            return {}
    
    def _fetch_cosmic_variant_info(self, cosmic_id: str) -> Dict[str, Any]:
        """Fetch variant information from COSMIC"""
        # COSMIC API requires authentication
        # This is a placeholder implementation
        
        return {
            'source': 'COSMIC',
            'note': 'COSMIC API access requires authentication'
        }
    
    def _fetch_hgvs_variant_info(self, hgvs_notation: str) -> Dict[str, Any]:
        """Fetch variant information using HGVS notation"""
        # Implementation would depend on available HGVS parsers
        # This is a placeholder
        
        return {
            'hgvs_notation': hgvs_notation,
            'parsed': self._parse_hgvs_notation(hgvs_notation)
        }
    
    def _parse_hgvs_notation(self, hgvs: str) -> Dict[str, Any]:
        """Parse HGVS notation into components"""
        patterns = {
            'coding': r'c\.(\d+)([ACGTUacgtu_]+)?(?:>|del|ins|dup|inv)([ACGTUacgtu_]*)',
            'protein': r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})?',
            'genomic': r'g\.(\d+)([ACGTUacgtu_]+)?(?:>|del|ins|dup|inv)([ACGTUacgtu_]*)'
        }
        
        for notation_type, pattern in patterns.items():
            match = re.match(pattern, hgvs)
            if match:
                return {
                    'type': notation_type,
                    'components': match.groups()
                }
        
        return {'type': 'unknown'}
    
    def _fetch_clinvar_variant_info(self, variant_id: str) -> Dict[str, Any]:
        """Fetch clinical significance from ClinVar"""
        try:
            url = f"{self.databases['clinvar']}esearch.fcgi"
            params = {
                'db': 'clinvar',
                'term': variant_id,
                'retmode': 'json',
                'retmax': 1
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            count = int(data.get('esearchresult', {}).get('count', 0))
            
            if count > 0:
                clinvar_id = data['esearchresult']['idlist'][0]
                
                # Fetch summary
                summary_url = f"{self.databases['clinvar']}esummary.fcgi"
                summary_params = {
                    'db': 'clinvar',
                    'id': clinvar_id,
                    'retmode': 'json'
                }
                
                summary_response = self._make_request(summary_url, summary_params)
                summary_data = summary_response.json()
                
                result = summary_data.get('result', {}).get(clinvar_id, {})
                
                return {
                    'clinical_significance': result.get('clinical_significance', ''),
                    'review_status': result.get('review_status', ''),
                    'conditions': result.get('conditions', []),
                    'gene': result.get('gene', {})
                }
        
        except Exception as e:
            logger.debug(f"Error fetching ClinVar info for {variant_id}: {e}")
        
        return {}
    
    def _make_request(self, url: str, params: Dict, retry_count: int = 0) -> requests.Response:
        """Make HTTP request with rate limiting and retry logic"""
        # Rate limiting
        elapsed = (datetime.now() - self.last_request_time).total_seconds()
        if elapsed < self.request_delay:
            asyncio.sleep(self.request_delay - elapsed)
        
        try:
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            self.last_request_time = datetime.now()
            
            if response.status_code == 429:  # Rate limited
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Rate limited, retrying in {wait_time} seconds...")
                    asyncio.sleep(wait_time)
                    return self._make_request(url, params, retry_count + 1)
            
            response.raise_for_status()
            return response
        
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.info(f"Request failed, retrying in {wait_time} seconds...")
                asyncio.sleep(wait_time)
                return self._make_request(url, params, retry_count + 1)
            raise
    
    def _get_from_cache(self, key: str) -> Any:
        """Get data from cache"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.pkl.gz"
        
        if cache_file.exists():
            try:
                # Check if cache is expired
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - mtime < self.cache_expiry:
                    with gzip.open(cache_file, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                logger.debug(f"Error reading cache {key}: {e}")
        
        return None
    
    def _save_to_cache(self, key: str, data: Any):
        """Save data to cache"""
        if not self.cache_enabled:
            return
        
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.pkl.gz"
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving to cache {key}: {e}")
    
    def _cache_entities(self, text: str, entities: Dict):
        """Cache extracted entities for future reference"""
        cache_key = f"entity_extraction_{hashlib.md5(text.encode()).hexdigest()}"
        self._save_to_cache(cache_key, entities)
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document file and extract genetic information.
        
        Args:
            file_path: Path to document file
        
        Returns:
            Dictionary with processed information
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read file based on extension
            if path.suffix.lower() == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif path.suffix.lower() == '.pdf':
                content = self._extract_text_from_pdf(path)
            elif path.suffix.lower() in ['.doc', '.docx']:
                content = self._extract_text_from_docx(path)
            elif path.suffix.lower() in ['.json', '.xml', '.csv']:
                content = self._extract_text_from_structured(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # Extract entities
            entities = self.extract_genetic_entities(content, validate=True)
            
            # Analyze document
            analysis = {
                'file_name': path.name,
                'file_size': path.stat().st_size,
                'content_length': len(content),
                'word_count': len(content.split()),
                'entity_counts': entities['metadata']['entity_counts'],
                'extraction_time': entities['metadata']['extraction_time'],
                'timestamp': datetime.now().isoformat()
            }
            
            result = {
                'analysis': analysis,
                'entities': entities,
                'content_preview': content[:500] + '...' if len(content) > 500 else content
            }
            
            logger.info(f"Processed document {path.name}: {analysis['entity_counts']}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {'error': str(e)}
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text
        
        except ImportError:
            logger.warning("PyPDF2 not installed, cannot extract text from PDF")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _extract_text_from_docx(self, docx_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            import docx
            
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        except ImportError:
            logger.warning("python-docx not installed, cannot extract text from DOCX")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    def _extract_text_from_structured(self, file_path: Path) -> str:
        """Extract text from structured files (JSON, XML, CSV)"""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return json.dumps(data, indent=2)
            
            elif suffix == '.xml':
                tree = ET.parse(file_path)
                root = tree.getroot()
                return ET.tostring(root, encoding='unicode')
            
            elif suffix == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string()
            
            else:
                raise ValueError(f"Unsupported structured format: {suffix}")
        
        except Exception as e:
            logger.error(f"Error extracting text from structured file: {e}")
            return ""
    
    def batch_process(self, file_paths: List[str], parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            parallel: Whether to process in parallel
        
        Returns:
            List of processing results
        """
        results = []
        
        if parallel:
            # Parallel processing (simplified version)
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(self.process_document, file_path): file_path 
                    for file_path in file_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        results.append({'file': file_path, 'error': str(e)})
        else:
            # Sequential processing
            for file_path in file_paths:
                try:
                    result = self.process_document(file_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append({'file': file_path, 'error': str(e)})
        
        return results
    
    def export_results(self, results: List[Dict], output_format: str = 'json', 
                      output_path: Optional[str] = None) -> str:
        """
        Export processing results to file.
        
        Args:
            results: List of processing results
            output_format: Output format ('json', 'csv', 'excel')
            output_path: Output file path (optional)
        
        Returns:
            Path to exported file
        """
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"./data/exports/genetics_export_{timestamp}.{output_format}"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            if output_format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            elif output_format.lower() == 'csv':
                # Flatten results for CSV
                flat_results = []
                for result in results:
                    if 'error' in result:
                        flat_results.append({'file': result.get('file', ''), 'error': result['error']})
                    else:
                        flat_result = {
                            'file': result['analysis']['file_name'],
                            'word_count': result['analysis']['word_count'],
                            'gene_count': result['analysis']['entity_counts'].get('genes', 0),
                            'disease_count': result['analysis']['entity_counts'].get('diseases', 0),
                            'extraction_time': result['analysis']['extraction_time']
                        }
                        flat_results.append(flat_result)
                
                df = pd.DataFrame(flat_results)
                df.to_csv(output_path, index=False, encoding='utf-8')
            
            elif output_format.lower() in ['excel', 'xlsx']:
                # Create Excel workbook with multiple sheets
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = []
                    for result in results:
                        if 'error' not in result:
                            summary_data.append({
                                'File': result['analysis']['file_name'],
                                'Genes': result['analysis']['entity_counts'].get('genes', 0),
                                'Diseases': result['analysis']['entity_counts'].get('diseases', 0),
                                'Variants': result['analysis']['entity_counts'].get('variants', 0),
                                'Extraction Time (s)': result['analysis']['extraction_time']
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Detailed entities sheet
                    entities_data = []
                    for result in results:
                        if 'error' not in result and 'entities' in result:
                            for gene in result['entities'].get('genes', []):
                                if isinstance(gene, dict):
                                    gene_name = gene.get('symbol', '')
                                else:
                                    gene_name = gene
                                
                                entities_data.append({
                                    'File': result['analysis']['file_name'],
                                    'Entity Type': 'Gene',
                                    'Entity': gene_name,
                                    'Count': 1
                                })
                    
                    if entities_data:
                        entities_df = pd.DataFrame(entities_data)
                        entities_df.to_excel(writer, sheet_name='Entities', index=False)
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            logger.info(f"Exported {len(results)} results to {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl.gz"))
        
        return {
            'cache_size': len(cache_files),
            'entity_dictionary_sizes': {
                'genes': len(self.entity_dictionaries['genes']),
                'diseases': len(self.entity_dictionaries['diseases']),
                'proteins': len(self.entity_dictionaries['proteins']),
                'chemicals': len(self.entity_dictionaries['chemicals'])
            },
            'cache_dir': str(self.cache_dir),
            'databases': list(self.databases.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_cache(self, cache_type: str = 'all') -> bool:
        """
        Clear cache.
        
        Args:
            cache_type: Type of cache to clear ('all', 'validation', 'gene_info', 'variant_info')
        
        Returns:
            Success status
        """
        try:
            if cache_type == 'all':
                cache_files = list(self.cache_dir.glob("*.pkl.gz"))
            else:
                cache_files = list(self.cache_dir.glob(f"*{cache_type}*.pkl.gz"))
            
            for cache_file in cache_files:
                cache_file.unlink()
            
            logger.info(f"Cleared {len(cache_files)} cache files")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


# Example usage and testing
def main():
    """Test the genetics data processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Genetics Data Processor')
    parser.add_argument('--extract', type=str, help='Extract entities from text')
    parser.add_argument('--validate-gene', type=str, help='Validate a gene symbol')
    parser.add_argument('--gene-info', type=str, help='Get gene information')
    parser.add_argument('--variant-info', type=str, help='Get variant information')
    parser.add_argument('--process-file', type=str, help='Process a document file')
    parser.add_argument('--batch-process', nargs='+', help='Process multiple files')
    parser.add_argument('--stats', action='store_true', help='Show processor statistics')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache')
    
    args = parser.parse_args()
    
    processor = GeneticsDataProcessor()
    
    if args.extract:
        print(f"\nExtracting entities from: {args.extract}")
        entities = processor.extract_genetic_entities(args.extract, validate=True)
        print(json.dumps(entities, indent=2))
    
    elif args.validate_gene:
        print(f"\nValidating gene: {args.validate_gene}")
        validation = processor.validate_gene(args.validate_gene)
        print(json.dumps(validation, indent=2))
    
    elif args.gene_info:
        print(f"\nFetching info for gene: {args.gene_info}")
        gene_info = processor.fetch_gene_info(args.gene_info, detailed=True)
        print(json.dumps(gene_info, indent=2))
    
    elif args.variant_info:
        print(f"\nFetching info for variant: {args.variant_info}")
        variant_info = processor.fetch_variant_info(args.variant_info)
        print(json.dumps(variant_info, indent=2))
    
    elif args.process_file:
        print(f"\nProcessing file: {args.process_file}")
        result = processor.process_document(args.process_file)
        print(json.dumps(result, indent=2))
    
    elif args.batch_process:
        print(f"\nBatch processing {len(args.batch_process)} files")
        results = processor.batch_process(args.batch_process, parallel=True)
        
        for result in results:
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Processed {result['analysis']['file_name']}: "
                      f"{result['analysis']['entity_counts']}")
        
        # Export results
        export_path = processor.export_results(results, 'json')
        print(f"\nResults exported to: {export_path}")
    
    elif args.stats:
        stats = processor.get_statistics()
        print("\nProcessor Statistics:")
        print(json.dumps(stats, indent=2))
    
    elif args.clear_cache:
        processor.clear_cache('all')
        print("Cache cleared")
    
    else:
        # Interactive mode
        print("Genetics Data Processor - Interactive Mode")
        print("Commands: quit, stats, clear, help")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nEnter text to extract entities: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'stats':
                    stats = processor.get_statistics()
                    print(json.dumps(stats, indent=2))
                elif user_input.lower() == 'clear':
                    processor.clear_cache('all')
                    print("Cache cleared")
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  <text> - Extract entities from text")
                    print("  stats - Show processor statistics")
                    print("  clear - Clear cache")
                    print("  quit - Exit")
                elif user_input:
                    entities = processor.extract_genetic_entities(user_input, validate=True)
                    print(f"\nExtracted {sum(entities['metadata']['entity_counts'].values())} entities:")
                    
                    for entity_type, entity_list in entities.items():
                        if entity_type != 'metadata' and entity_list:
                            if isinstance(entity_list, list):
                                print(f"  {entity_type}: {entity_list}")
                            elif isinstance(entity_list, dict):
                                print(f"  {entity_type}:")
                                for sub_type, sub_list in entity_list.items():
                                    if sub_list:
                                        print(f"    {sub_type}: {sub_list}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
