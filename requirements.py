# Core Flask dependencies
flask==2.3.3
flask-cors==4.0.0

# PyTorch with CPU-only version for better compatibility
torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu
# Alternative if above doesn't work: torch==2.1.2

# Transformers and related dependencies
transformers==4.35.2
accelerate==0.24.1

# Supporting libraries with stable versions
numpy==1.24.3
requests==2.31.0
protobuf==3.20.3

# Text processing
sentencepiece==0.1.99
tokenizers==0.15.0

# Optional: for better performance
psutil==5.9.5
