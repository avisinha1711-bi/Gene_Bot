import json
import re
import os
from datetime import datetime

class GeneticsChatbot:
    def __init__(self, knowledge_file="data/genetics_knowledge.json"):
        self.knowledge_file = knowledge_file
        self.knowledge_base = self.load_knowledge_base()
        self.conversation_history = []
        
    def load_knowledge_base(self):
        """Load genetics knowledge from JSON file."""
        default_knowledge = {
            "concepts": {
                "dna": "DNA (Deoxyribonucleic Acid) is the hereditary material in humans and almost all other organisms. It contains the genetic instructions for development, functioning, growth, and reproduction.",
                "rna": "RNA (Ribonucleic Acid) is a nucleic acid that plays crucial roles in coding, decoding, regulation, and expression of genes. Major types include mRNA, tRNA, and rRNA.",
                "gene": "A gene is the basic physical and functional unit of heredity. Genes are made up of DNA and act as instructions to make molecules called proteins.",
                "chromosome": "Chromosomes are thread-like structures located inside the nucleus of animal and plant cells. Each chromosome is made of protein and a single molecule of DNA.",
                "mutation": "A mutation is a change in a DNA sequence. Mutations can result from DNA copying mistakes, exposure to radiation or chemicals, or viral infection.",
                "protein": "Proteins are large, complex molecules that play many critical roles in the body. They do most of the work in cells and are required for the structure, function, and regulation of the body's tissues and organs.",
                "transcription": "Transcription is the process of making an RNA copy of a gene's DNA sequence. This copy, called mRNA, carries the genetic information needed to make proteins.",
                "translation": "Translation is the process where ribosomes in the cytoplasm or ER synthesize proteins after transcription of DNA to RNA in the cell's nucleus.",
                "genotype": "Genotype refers to the genetic constitution of an individual organism. It represents the specific allele combination for a particular gene.",
                "phenotype": "Phenotype refers to the observable physical properties of an organism including appearance, development, and behavior.",
                "allele": "An allele is one of two or more versions of a gene. An individual inherits two alleles for each gene, one from each parent.",
                "heredity": "Heredity is the passing of genetic traits from parents to their offspring through sexual or asexual reproduction.",
                "genome": "A genome is an organism's complete set of DNA, including all of its genes. The human genome contains about 3 billion base pairs.",
                "mitosis": "Mitosis is a process of cell division that results in two genetically identical daughter cells developing from a single parent cell.",
                "meiosis": "Meiosis is a special type of cell division that produces gametes (sperm and egg cells) with half the number of chromosomes of the parent cell."
            },
            "patterns": {
                "what_is": ["what is", "tell me about", "explain", "define"],
                "how_does": ["how does", "how do", "how is", "how are"],
                "difference": ["difference between", "different from", "compare"],
                "examples": ["example of", "examples of", "give me an example"]
            }
        }
        
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.knowledge_file), exist_ok=True)
                with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                    json.dump(default_knowledge, f, indent=2)
                return default_knowledge
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {e}")
            return default_knowledge
    
    def get_response(self, user_input):
        """Generate a response to user input."""
        user_input = user_input.lower().strip()
        
        # Log the conversation
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": ""
        })
        
        response = self._generate_response(user_input)
        
        # Update conversation history with response
        self.conversation_history[-1]["bot"] = response
        
        return response
    
    def _generate_response(self, user_input):
        """Generate response based on user input patterns."""
        # Check for greetings
        if any(word in user_input for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm your genetics assistant. I can help you understand DNA, RNA, genes, mutations, and other genetics concepts. What would you like to know?"
        
        # Check for thanks
        if any(word in user_input for word in ['thank', 'thanks', 'appreciate']):
            return "You're welcome! I'm happy to help with your genetics questions. Is there anything else you'd like to know?"
        
        # Direct concept lookup
        for concept, definition in self.knowledge_base["concepts"].items():
            if concept in user_input:
                return definition
        
        # Pattern-based responses
        if any(pattern in user_input for pattern in self.knowledge_base["patterns"]["what_is"]):
            for concept in self.knowledge_base["concepts"]:
                if concept in user_input:
                    return self.knowledge_base["concepts"][concept]
            return "I can explain various genetics concepts like DNA, RNA, genes, chromosomes, mutations, proteins, transcription, translation, genotype, phenotype, alleles, heredity, genome, mitosis, and meiosis. Which specific concept would you like me to explain?"
        
        elif any(pattern in user_input for pattern in self.knowledge_base["patterns"]["difference"]):
            if "dna and rna" in user_input or "rna and dna" in user_input:
                return "DNA vs RNA: DNA is double-stranded and contains deoxyribose sugar, while RNA is single-stranded and contains ribose sugar. DNA uses thymine (T) as a base, while RNA uses uracil (U). DNA stores genetic information long-term, while RNA acts as a messenger and plays various roles in protein synthesis."
            elif "genotype and phenotype" in user_input:
                return "Genotype vs Phenotype: Genotype refers to the genetic makeup (specific alleles) of an organism, while phenotype refers to the observable physical characteristics. Genotype influences phenotype, but environmental factors can also affect phenotypic expression."
            elif "mitosis and meiosis" in user_input:
                return "Mitosis vs Meiosis: Mitosis produces two identical diploid daughter cells for growth and repair, while meiosis produces four genetically diverse haploid gametes for sexual reproduction. Meiosis involves two divisions and genetic recombination through crossing over."
        
        elif any(pattern in user_input for pattern in self.knowledge_base["patterns"]["how_does"]):
            if "transcription work" in user_input:
                return "Transcription works in three main steps: 1) Initiation - RNA polymerase binds to promoter region of DNA, 2) Elongation - RNA polymerase builds mRNA complementary to the DNA template strand, 3) Termination - mRNA transcript is released and RNA polymerase detaches from DNA."
            elif "translation work" in user_input:
                return "Translation works by ribosomes reading mRNA codons and assembling amino acids into proteins: 1) Initiation - ribosome assembles around start codon, 2) Elongation - tRNA brings amino acids, ribosome forms peptide bonds, 3) Termination - stop codon signals release of completed protein."
            elif "dna replication work" in user_input:
                return "DNA replication is semi-conservative and involves: 1) Helicase unwinds DNA double helix, 2) Primase adds RNA primers, 3) DNA polymerase adds complementary nucleotides, 4) Leading strand is continuous, lagging strand is in Okazaki fragments, 5) Ligase joins fragments together."
        
        # Default response for unknown queries
        return "I'm not sure I understand your genetics question. I can help explain concepts like DNA, RNA, genes, mutations, protein synthesis, inheritance patterns, and cellular division. Could you try rephrasing your question?"
    
    def save_conversation_history(self, filename="conversation_history.json"):
        """Save conversation history to a file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving conversation history: {e}")
            return False
