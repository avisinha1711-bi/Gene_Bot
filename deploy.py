# deploy.py
"""
Deployment script for GeneBot.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def check_dependencies():
    """Check and install required dependencies"""
    print("Checking dependencies...")
    
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    else:
        print("requirements.txt not found!")

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "models",
        "knowledge_base",
        "templates",
        "static",
        "data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}/")

def download_model():
    """Download Mistral-7B model"""
    print("\nDownloading Mistral-7B model...")
    
    model_dir = "models/mistral-7b"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
        # For production, you might want to use a smaller model or pre-download
        print("Note: For production, consider using a smaller model or")
        print("pre-downloading the model due to size (~14GB).")
        print("\nTo download manually:")
        print("from transformers import AutoModelForCausalLM, AutoTokenizer")
        print('model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")')
        print('model.save_pretrained("./models/mistral-7b")')
    
    return model_dir

def create_sample_knowledge():
    """Create sample knowledge base"""
    print("\nCreating sample knowledge base...")
    
    kb_dir = "knowledge_base"
    sample_data = {
        "topics": ["DNA Basics", "CRISPR Technology", "Genetic Inheritance"],
        "documents": [
            "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions.",
            "CRISPR is a revolutionary gene-editing technology.",
            "Genetic inheritance follows Mendelian principles of dominance and recessiveness."
        ]
    }
    
    import json
    with open(os.path.join(kb_dir, "sample_knowledge.json"), "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample knowledge base created!")

def setup_environment():
    """Set up environment variables"""
    print("\nSetting up environment...")
    
    env_content = """# GeneBot Configuration
MODEL_PATH=models/mistral-7b
KNOWLEDGE_BASE_PATH=knowledge_base
USE_RAG=true
DEBUG=false

# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=production
PORT=5000
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("Environment file created: .env")

def create_procfile():
    """Create Procfile for deployment"""
    print("\nCreating Procfile...")
    
    with open("Procfile", "w") as f:
        f.write("web: gunicorn app:app --bind 0.0.0.0:$PORT\n")
    
    print("Procfile created!")

def test_deployment():
    """Test if deployment is ready"""
    print("\nTesting deployment...")
    
    # Test imports
    try:
        from mistral_inference import MistralGeneticsBot
        print("✓ mistral_inference.py imports successfully")
    except Exception as e:
        print(f"✗ Error importing mistral_inference: {e}")
    
    # Test Flask app
    try:
        from app import app
        print("✓ Flask app imports successfully")
    except Exception as e:
        print(f"✗ Error importing Flask app: {e}")
    
    # Check if all required files exist
    required_files = [
        "app.py",
        "config.py",
        "knowledge_base.py",
        "mistral_inference.py",
        "requirements.txt",
        "runtime.txt"
    ]
    
    print("\nChecking required files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (MISSING)")

def main():
    """Main deployment setup"""
    print("=" * 60)
    print("GENE BOT DEPLOYMENT SETUP")
    print("=" * 60)
    
    try:
        # Step 1: Check dependencies
        check_dependencies()
        
        # Step 2: Create directories
        create_directories()
        
        # Step 3: Download model (note about size)
        download_model()
        
        # Step 4: Create sample knowledge
        create_sample_knowledge()
        
        # Step 5: Setup environment
        setup_environment()
        
        # Step 6: Create Procfile
        create_procfile()
        
        # Step 7: Test deployment
        test_deployment()
        
        print("\n" + "=" * 60)
        print("DEPLOYMENT SETUP COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Download the model manually (see instructions above)")
        print("2. Add your genetics data to 'knowledge_base/' directory")
        print("3. Run training: python train_model.py --train --data your_data.json")
        print("4. Start the server: python app.py")
        print("5. Visit: http://localhost:5000")
        print("\nFor Heroku deployment:")
        print("  heroku create your-genebot-app")
        print("  git push heroku main")
        
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
