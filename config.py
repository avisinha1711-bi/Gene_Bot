# Genetics Database APIs
PUBMED_API_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_BASE = "https://eutils.ncbi.nlm.nih.gov"
ENSEMBL_API_BASE = "https://rest.ensembl.org"

# API Endpoints
API_ENDPOINTS = {
    'pubmed_search': f"{PUBMED_API_BASE}/esearch.fcgi",
    'pubmed_fetch': f"{PUBMED_API_BASE}/efetch.fcgi",
    'ncbi_gene': "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
    'ensembl_gene': f"{ENSEMBL_API_BASE}/lookup/symbol/human/{{gene_symbol}}"
}

# Knowledge Base Settings
KNOWLEDGE_BASE_CONFIG = {
    'max_conversation_history': 100,
    'learning_rate': 0.1,
    'feedback_threshold': 0.8,
    'auto_update_hours': 24
}

# Genetics Databases
GENETICS_DATABASES = {
    'pubmed': 'PubMed Central',
    'ncbi_gene': 'NCBI Gene Database',
    'ensembl': 'Ensembl Genome Browser',
    'clinvar': 'ClinVar Clinical Variants',
    'omim': 'Online Mendelian Inheritance in Man'
}
