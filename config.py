# Model Configuration
MODEL_CONFIG = {
    'model_name': 'microsoft/DialoGPT-small',
    'max_length': 512,
    'max_new_tokens': 150,
    'temperature': 0.7,
    'top_p': 0.9,
    'repetition_penalty': 1.1,
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False
}

# Genetics-specific knowledge base
GENETICS_KNOWLEDGE = {
    "core_concepts": [
        "DNA structure and replication",
        "RNA transcription and processing", 
        "Protein synthesis and translation",
        "Genetic inheritance patterns",
        "Gene regulation and expression",
        "Mutation types and effects",
        "Chromosomal abnormalities",
        "Molecular biology techniques"
    ],
    "key_terms": [
        "genome", "chromosome", "gene", "allele", "genotype", "phenotype",
        "transcription", "translation", "mutation", "replication",
        "mitosis", "meiosis", "heredity", "evolution"
    ]
}

# Response templates for fallback
RESPONSE_TEMPLATES = {
    "greeting": "Hello! I'm GeneBot, your intelligent genetics assistant. I can help you understand complex genetics concepts, analyze genetic patterns, and explain molecular biology processes. What would you like to know?",
    "fallback": "I'm not quite sure about that specific genetics question. Could you provide more context or ask about DNA, RNA, genes, mutations, protein synthesis, or other genetics topics?",
    "technical_error": "I'm experiencing some technical difficulties. Please try again in a moment."
}
