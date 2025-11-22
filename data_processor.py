import re
import requests
from bs4 import BeautifulSoup

class GeneticsDataProcessor:
    def __init__(self):
        self.gene_pattern = re.compile(r'[A-Z0-9]+[A-Z][0-9]*')
        self.disease_pattern = re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)* syndrome|disease', re.IGNORECASE)
    
    def extract_genetic_entities(self, text):
        """Extract genes, diseases, and variants from text"""
        entities = {
            'genes': self.gene_pattern.findall(text),
            'diseases': self.disease_pattern.findall(text),
            'variants': self._extract_variants(text)
        }
        return entities
    
    def _extract_variants(self, text):
        """Extract genetic variants (rs numbers, chromosomal positions)"""
        rs_pattern = re.compile(r'rs\d+', re.IGNORECASE)
        chr_pattern = re.compile(r'chr\d+[:\-]\d+', re.IGNORECASE)
        
        return {
            'rs_ids': rs_pattern.findall(text),
            'chromosomal_positions': chr_pattern.findall(text)
        }
    
    def fetch_gene_info(self, gene_symbol):
        """Fetch detailed gene information from NCBI"""
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'gene',
            'term': f"{gene_symbol}[Gene Name] AND human[Organism]",
            'retmode': 'json'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            return data
        except Exception as e:
            print(f"Error fetching gene info: {e}")
            return None
