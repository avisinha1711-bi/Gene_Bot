# train_model.py
"""
Train a genetics-focused chatbot model for Gene_Bot.
This script prepares the data, trains the model, and saves it for deployment.
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class GeneticsChatbotTrainer:
    def __init__(self, config=None):
        """Initialize the trainer with configuration."""
        self.config = config or {
            'max_words': 5000,
            'max_len': 50,
            'embedding_dim': 128,
            'lstm_units': 64,
            'dropout_rate': 0.3,
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2,
            'test_size': 0.15
        }
        
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.classes = None
        
    def load_sample_data(self):
        """
        Load sample genetics data.
        In a real scenario, you would load from your actual dataset.
        """
        # Sample genetics-focused Q&A pairs
        genetics_data = {
            "questions": [
                "What is DNA?",
                "Explain gene expression",
                "What are chromosomes?",
                "How does DNA replication work?",
                "What is RNA?",
                "Explain Mendelian inheritance",
                "What are mutations?",
                "How do genes affect traits?",
                "What is genetic engineering?",
                "Explain CRISPR technology",
                "What is genetic variation?",
                "How does transcription work?",
                "What are alleles?",
                "Explain protein synthesis",
                "What is a genome?",
                "How do cells divide?",
                "What is genetic counseling?",
                "Explain epigenetics",
                "What are genetic disorders?",
                "How does DNA sequencing work?"
            ],
            "answers": [
                "DNA (Deoxyribonucleic Acid) is the hereditary material in humans and almost all other organisms. It contains the biological instructions that make each species unique.",
                "Gene expression is the process by which information from a gene is used to synthesize a functional gene product, typically proteins or functional RNA molecules.",
                "Chromosomes are thread-like structures located inside the nucleus of animal and plant cells, made of protein and a single molecule of DNA.",
                "DNA replication is the biological process of producing two identical replicas of DNA from one original DNA molecule.",
                "RNA (Ribonucleic Acid) is a nucleic acid present in all living cells that acts as a messenger carrying instructions from DNA for controlling protein synthesis.",
                "Mendelian inheritance refers to patterns of inheritance for traits controlled by single genes with two alleles, following Mendel's laws.",
                "Mutations are changes in the DNA sequence that can affect gene function and may lead to genetic disorders or variations.",
                "Genes contain instructions for making proteins that determine traits through complex interactions with environment and other genes.",
                "Genetic engineering is the direct manipulation of an organism's genes using biotechnology to alter its characteristics.",
                "CRISPR is a gene-editing technology that allows precise modification of DNA sequences in living organisms.",
                "Genetic variation refers to differences in DNA sequences among individuals within a population.",
                "Transcription is the first step of gene expression where DNA is copied into RNA by the enzyme RNA polymerase.",
                "Alleles are different versions of the same gene that arise by mutation and are found at the same locus on chromosomes.",
                "Protein synthesis is the process by which cells build proteins based on genetic instructions in DNA, involving transcription and translation.",
                "A genome is the complete set of genetic material (DNA or RNA) in an organism.",
                "Cell division is the process by which a parent cell divides into two or more daughter cells, important for growth and repair.",
                "Genetic counseling is the process of helping people understand and adapt to medical, psychological aspects of genetic contributions to disease.",
                "Epigenetics studies heritable changes in gene expression that do not involve changes to the underlying DNA sequence.",
                "Genetic disorders are diseases caused by abnormalities in an individual's DNA, which can be inherited or occur spontaneously.",
                "DNA sequencing is the process of determining the precise order of nucleotides within a DNA molecule."
            ],
            "categories": [
                "basics", "expression", "structures", "processes", "basics",
                "inheritance", "mutations", "traits", "engineering", "technology",
                "variation", "processes", "genetics", "processes", "basics",
                "cell_biology", "applications", "epigenetics", "disorders", "technology"
            ]
        }
        
        return pd.DataFrame(genetics_data)
    
    def load_custom_data(self, filepath):
        """
        Load custom data from JSON or CSV file.
        Supports multiple formats for flexibility.
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
    
    def preprocess_text(self, texts, is_train=True):
        """Preprocess and tokenize text data."""
        if is_train:
            self.tokenizer = Tokenizer(
                num_words=self.config['max_words'],
                oov_token='<OOV>',
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.config['max_len'], padding='post')
        
        return padded
    
    def encode_labels(self, labels):
        """Encode categorical labels."""
        from sklearn.preprocessing import LabelEncoder
        
        self.label_encoder = LabelEncoder()
        encoded = self.label_encoder.fit_transform(labels)
        self.classes = self.label_encoder.classes_
        
        # Convert to one-hot encoding
        return tf.keras.utils.to_categorical(encoded, num_classes=len(self.classes))
    
    def build_model(self, num_classes):
        """Build the neural network model."""
        model = Sequential([
            Embedding(
                input_dim=self.config['max_words'],
                output_dim=self.config['embedding_dim'],
                input_length=self.config['max_len']
            ),
            Bidirectional(LSTM(self.config['lstm_units'], return_sequences=True)),
            Dropout(self.config['dropout_rate']),
            LSTM(self.config['lstm_units'] // 2),
            Dropout(self.config['dropout_rate']),
            Dense(128, activation='relu'),
            Dropout(self.config['dropout_rate'] / 2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data_path=None, use_sample=True):
        """
        Main training pipeline.
        
        Args:
            data_path: Path to custom data file (JSON or CSV)
            use_sample: Whether to use sample data if no custom data provided
        """
        print("=" * 60)
        print("GENETICS CHATBOT MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        if data_path:
            print(f"Loading custom data from: {data_path}")
            df = self.load_custom_data(data_path)
        elif use_sample:
            print("Loading sample genetics data...")
            df = self.load_sample_data()
        else:
            raise ValueError("No data provided for training.")
        
        print(f"Dataset size: {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Prepare data
        X = self.preprocess_text(df['questions'].tolist(), is_train=True)
        y = self.encode_labels(df['categories'].tolist())
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=42,
            stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Number of classes: {len(self.classes)}")
        
        # Build model
        self.model = self.build_model(len(self.classes))
        
        print("\nModel architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("\n" + "=" * 60)
        print("Training started...")
        print("=" * 60)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Training Accuracy:   {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy:       {test_acc:.4f}")
        
        return history
    
    def save_model(self, base_path='models'):
        """Save the trained model and related artifacts."""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(base_path, 'genetics_chatbot_model.keras')
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save tokenizer
        tokenizer_path = os.path.join(base_path, 'tokenizer.pickle')
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved to: {tokenizer_path}")
        
        # Save label encoder
        encoder_path = os.path.join(base_path, 'label_encoder.pickle')
        with open(encoder_path, 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Label encoder saved to: {encoder_path}")
        
        # Save classes
        classes_path = os.path.join(base_path, 'classes.npy')
        np.save(classes_path, self.classes)
        print(f"Classes saved to: {classes_path}")
        
        # Save config
        config_path = os.path.join(base_path, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved to: {config_path}")
        
        # Save sample responses for deployment
        self.save_response_templates(base_path)
    
    def save_response_templates(self, base_path):
        """Save response templates for different categories."""
        response_templates = {
            "basics": {
                "examples": ["What is DNA?", "What is RNA?", "What is a genome?"],
                "responses": [
                    "DNA is the hereditary material in organisms...",
                    "RNA is a nucleic acid that helps in protein synthesis...",
                    "A genome is the complete set of genetic material..."
                ]
            },
            "processes": {
                "examples": ["How does transcription work?", "Explain protein synthesis"],
                "responses": [
                    "Transcription is the process of copying DNA to RNA...",
                    "Protein synthesis involves transcription and translation..."
                ]
            }
        }
        
        templates_path = os.path.join(base_path, 'response_templates.json')
        with open(templates_path, 'w') as f:
            json.dump(response_templates, f, indent=2)
        print(f"Response templates saved to: {templates_path}")
    
    def predict(self, question):
        """Make a prediction for a single question."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Please train or load a model first.")
        
        # Preprocess question
        processed = self.preprocess_text([question], is_train=False)
        
        # Make prediction
        prediction = self.model.predict(processed, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        category = self.label_encoder.inverse_transform([class_idx])[0]
        
        return {
            'question': question,
            'category': category,
            'confidence': float(confidence),
            'all_predictions': prediction[0].tolist()
        }

def create_sample_dataset(output_path='data/sample_genetics_data.json'):
    """Create a sample dataset for testing."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sample_data = {
        "questions": [
            "What is DNA?",
            "Explain gene expression",
            "What are chromosomes?",
            "How does DNA replication work?",
            "What is RNA?",
            "Explain Mendelian inheritance",
            "What are mutations?",
            "How do genes affect traits?",
            "What is genetic engineering?",
            "Explain CRISPR technology"
        ],
        "answers": [
            "DNA (Deoxyribonucleic Acid) is the hereditary material...",
            "Gene expression is the process by which information from a gene...",
            "Chromosomes are thread-like structures located inside the nucleus...",
            "DNA replication is the biological process of producing two identical replicas...",
            "RNA (Ribonucleic Acid) is a nucleic acid present in all living cells...",
            "Mendelian inheritance refers to patterns of inheritance for traits...",
            "Mutations are changes in the DNA sequence that can affect gene function...",
            "Genes contain instructions for making proteins that determine traits...",
            "Genetic engineering is the direct manipulation of an organism's genes...",
            "CRISPR is a gene-editing technology that allows precise modification..."
        ],
        "categories": [
            "basics", "expression", "structures", "processes", "basics",
            "inheritance", "mutations", "traits", "engineering", "technology"
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample dataset created at: {output_path}")
    return output_path

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train genetics chatbot model')
    parser.add_argument('--data', type=str, help='Path to training data (JSON/CSV)')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for saved models')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset()
        return
    
    # Update config with command line arguments
    config = {
        'max_words': 5000,
        'max_len': 50,
        'embedding_dim': 128,
        'lstm_units': 64,
        'dropout_rate': 0.3,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'validation_split': 0.2,
        'test_size': 0.15
    }
    
    # Initialize trainer
    trainer = GeneticsChatbotTrainer(config)
    
    # Train model
    try:
        history = trainer.train(data_path=args.data, use_sample=(args.data is None))
        
        # Save model
        trainer.save_model(args.output_dir)
        
        # Test prediction
        print("\n" + "=" * 60)
        print("Sample Predictions")
        print("=" * 60)
        
        test_questions = [
            "What is DNA?",
            "How does gene editing work?",
            "Explain genetic inheritance"
        ]
        
        for question in test_questions:
            result = trainer.predict(question)
            print(f"\nQuestion: {result['question']}")
            print(f"Predicted category: {result['category']}")
            print(f"Confidence: {result['confidence']:.2%}")
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model artifacts saved in: {args.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
