// Genetics Knowledge Base
const geneticsKnowledge = {
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
};

// Global variables
let currentNotes = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadNotes();
    loadStats();
    
    // Add Enter key support for chat
    document.getElementById('user-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Add Enter key support for note title
    document.getElementById('note-title').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            document.getElementById('note-content').focus();
        }
    });
});

// Chat functionality
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessageToChat('user', message);
    userInput.value = '';
    
    // Generate bot response
    const response = generateBotResponse(message);
    setTimeout(() => {
        addMessageToChat('bot', response);
    }, 500);
}

function generateBotResponse(userInput) {
    const input = userInput.toLowerCase().trim();
    
    // Check for greetings
    if (input.includes('hello') || input.includes('hi') || input.includes('hey')) {
        return "Hello! I'm GeneBot, your genetics assistant. I can help you understand DNA, RNA, genes, mutations, and other genetics concepts. What would you like to know?";
    }
    
    // Check for thanks
    if (input.includes('thank') || input.includes('thanks')) {
        return "You're welcome! I'm happy to help with your genetics questions. Is there anything else you'd like to know?";
    }
    
    // Direct concept lookup
    for (const [concept, definition] of Object.entries(geneticsKnowledge)) {
        if (input.includes(concept)) {
            return definition;
        }
    }
    
    // Pattern-based responses
    if (input.includes('what is') || input.includes('tell me about') || input.includes('explain') || input.includes('define')) {
        for (const concept in geneticsKnowledge) {
            if (input.includes(concept)) {
                return geneticsKnowledge[concept];
            }
        }
        return "I can explain various genetics concepts like DNA, RNA, genes, chromosomes, mutations, proteins, transcription, translation, genotype, phenotype, alleles, heredity, genome, mitosis, and meiosis. Which specific concept would you like me to explain?";
    }
    
    if (input.includes('difference between') || input.includes('different from') || input.includes('compare')) {
        if (input.includes('dna') && input.includes('rna')) {
            return "DNA vs RNA: DNA is double-stranded and contains deoxyribose sugar, while RNA is single-stranded and contains ribose sugar. DNA uses thymine (T) as a base, while RNA uses uracil (U). DNA stores genetic information long-term, while RNA acts as a messenger and plays various roles in protein synthesis.";
        }
        if (input.includes('genotype') && input.includes('phenotype')) {
            return "Genotype vs Phenotype: Genotype refers to the genetic makeup (specific alleles) of an organism, while phenotype refers to the observable physical characteristics. Genotype influences phenotype, but environmental factors can also affect phenotypic expression.";
        }
        if (input.includes('mitosis') && input.includes('meiosis')) {
            return "Mitosis vs Meiosis: Mitosis produces two identical diploid daughter cells for growth and repair, while meiosis produces four genetically diverse haploid gametes for sexual reproduction. Meiosis involves two divisions and genetic recombination through crossing over.";
        }
    }
    
    if (input.includes('how does') || input.includes('how do') || input.includes('how is')) {
        if (input.includes('transcription')) {
            return "Transcription works in three main steps: 1) Initiation - RNA polymerase binds to promoter region of DNA, 2) Elongation - RNA polymerase builds mRNA complementary to the DNA template strand, 3) Termination - mRNA transcript is released and RNA polymerase detaches from DNA.";
        }
        if (input.includes('translation')) {
            return "Translation works by ribosomes reading mRNA codons and assembling amino acids into proteins: 1) Initiation - ribosome assembles around start codon, 2) Elongation - tRNA brings amino acids, ribosome forms peptide bonds, 3) Termination - stop codon signals release of completed protein.";
        }
        if (input.includes('dna replication')) {
            return "DNA replication is semi-conservative and involves: 1) Helicase unwinds DNA double helix, 2) Primase adds RNA primers, 3) DNA polymerase adds complementary nucleotides, 4) Leading strand is continuous, lagging strand is in Okazaki fragments, 5) Ligase joins fragments together.";
        }
    }
    
    // Default response
    return "I'm not sure I understand your genetics question. I can help explain concepts like DNA, RNA, genes, mutations, protein synthesis, inheritance patterns, and cellular division. Could you try rephrasing your question?";
}

function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'GeneBot'}:</strong> ${escapeHtml(message)}`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setSuggestion(question) {
    document.getElementById('user-input').value = question;
    document.getElementById('user-input').focus();
}

// Notes functionality
function loadNotes() {
    const savedNotes = localStorage.getItem('geneticsNotes');
    if (savedNotes) {
        currentNotes = JSON.parse(savedNotes);
    } else {
        currentNotes = [];
    }
    displayNotes(currentNotes);
}

function saveNotes() {
    localStorage.setItem('geneticsNotes', JSON.stringify(currentNotes));
}

function displayNotes(notes) {
    const notesList = document.getElementById('notes-list');
    
    if (notes.length === 0) {
        notesList.innerHTML = '<p>No notes yet. Start by adding some notes above!</p>';
        return;
    }
    
    notesList.innerHTML = notes.map(note => `
        <div class="note-item">
            <div class="note-header">
                <div class="note-title">${escapeHtml(note.title)}</div>
                <div class="note-category">${escapeHtml(note.category)}</div>
            </div>
            <div class="note-content">${escapeHtml(note.content)}</div>
            <div class="note-footer">
                <span>${note.timestamp}</span>
                <button class="delete-note" onclick="deleteNote('${note.id}')">Delete</button>
            </div>
        </div>
    `).join('');
}

function addNote() {
    const title = document.getElementById('note-title').value.trim();
    const content = document.getElementById('note-content').value.trim();
    const category = document.getElementById('note-category').value.trim() || 'general';
    
    if (!title || !content) {
        alert('Please enter both title and content for the note.');
        return;
    }
    
    const newNote = {
        id: generateId(),
        title: title,
        content: content,
        category: category,
        timestamp: new Date().toLocaleString()
    };
    
    currentNotes.unshift(newNote);
    saveNotes();
    
    // Clear form
    document.getElementById('note-title').value = '';
    document.getElementById('note-content').value = '';
    document.getElementById('note-category').value = '';
    
    // Update display
    displayNotes(currentNotes);
    loadStats();
}

function searchNotes() {
    const query = document.getElementById('search-query').value.trim().toLowerCase();
    
    if (!query) {
        displayNotes(currentNotes);
        return;
    }
    
    const filteredNotes = currentNotes.filter(note => 
        note.title.toLowerCase().includes(query) ||
        note.content.toLowerCase().includes(query) ||
        note.category.toLowerCase().includes(query)
    );
    
    displayNotes(filteredNotes);
}

function clearSearch() {
    document.getElementById('search-query').value = '';
    displayNotes(currentNotes);
}

function deleteNote(noteId) {
    if (!confirm('Are you sure you want to delete this note?')) {
        return;
    }
    
    currentNotes = currentNotes.filter(note => note.id !== noteId);
    saveNotes();
    displayNotes(currentNotes);
    loadStats();
}

function loadStats() {
    const stats = {
        total_notes: currentNotes.length,
        categories: [],
        category_counts: {}
    };
    
    // Calculate categories
    currentNotes.forEach(note => {
        if (!stats.categories.includes(note.category)) {
            stats.categories.push(note.category);
            stats.category_counts[note.category] = 1;
        } else {
            stats.category_counts[note.category]++;
        }
    });
    
    displayStats(stats);
}

function displayStats(stats) {
    const statsDiv = document.getElementById('stats');
    
    let statsHTML = '<h3>📊 Note Statistics</h3>';
    statsHTML += `<div class="stat-item"><span>Total Notes:</span><span>${stats.total_notes}</span></div>`;
    
    if (stats.categories.length > 0) {
        statsHTML += '<div class="stat-item"><strong>Categories:</strong><span></span></div>';
        stats.categories.forEach(category => {
            const count = stats.category_counts[category] || 0;
            statsHTML += `<div class="stat-item"><span>• ${escapeHtml(category)}:</span><span>${count}</span></div>`;
        });
    }
    
    statsDiv.innerHTML = statsHTML;
}

// Utility functions
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
