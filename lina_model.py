# importation/downloading stuff

import pandas as pd
import numpy as np
import re
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ['scikit-learn', 'nltk', 'pandas', 'numpy', 'imblearn']
for package in packages:
    install_package(package)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK download failed, continuing without some preprocessing features")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class EnhancedIncidentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Combining FR/ENG stopwords
        try:
            self.stop_words = set(stopwords.words('english') + stopwords.words('french'))
        except:
            # Basic stopwords if NLTK download failed
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                                  'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'dans', 'sur', 'pour', 'de', 'avec'])
        
        # Hardware/Equipment keywords
        self.hardware_keywords = {
            'english': ['computer', 'laptop', 'desktop', 'monitor', 'screen', 'keyboard', 'mouse', 'printer', 
                       'headphones', 'speakers', 'microphone', 'camera', 'webcam', 'cable', 'charger', 
                       'hardware', 'equipment', 'device', 'machine', 'phone', 'tablet', 'router', 
                       'switch', 'server', 'disk', 'drive', 'memory', 'ram', 'cpu', 'processor'],
            'french': ['ordinateur', 'portable', 'bureau', 'moniteur', 'écran', 'clavier', 'souris', 'imprimante',
                      'casque', 'haut-parleurs', 'microphone', 'caméra', 'webcam', 'câble', 'chargeur',
                      'matériel', 'équipement', 'appareil', 'machine', 'téléphone', 'tablette', 'routeur',
                      'commutateur', 'serveur', 'disque', 'lecteur', 'mémoire', 'processeur']
        }
        
        # Application keywords
        self.application_keywords = {
            'english': ['software', 'application', 'app', 'program', 'system', 'website', 'browser', 
                       'outlook', 'excel', 'word', 'powerpoint', 'teams', 'zoom', 'chrome', 'firefox',
                       'install', 'update', 'upgrade', 'download', 'login', 'password', 'account'],
            'french': ['logiciel', 'application', 'programme', 'système', 'site web', 'navigateur',
                      'outlook', 'excel', 'word', 'powerpoint', 'teams', 'zoom', 'chrome', 'firefox',
                      'installer', 'mettre à jour', 'télécharger', 'connexion', 'mot de passe', 'compte']
        }
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        except:
            # Basic tokenization if NLTK fails
            tokens = [word for word in text.split() 
                     if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def add_domain_features(self, text):
        text_lower = text.lower()
        
        # Count hardware keywords
        hardware_count = sum(1 for word in self.hardware_keywords['english'] + self.hardware_keywords['french'] 
                           if word in text_lower)
        
        # Count application keywords
        app_count = sum(1 for word in self.application_keywords['english'] + self.application_keywords['french'] 
                       if word in text_lower)
        
        # Add feature indicators
        features = []
        if hardware_count > 0:
            features.append("HARDWARE_INDICATOR")
        if app_count > 0:
            features.append("APPLICATION_INDICATOR")
            
        return ' '.join(features)
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the CSV data with enhanced analysis"""
        print("Loading data...")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded data with {encoding} encoding")
                break
            except:
                continue
        
        if df is None:
            raise ValueError("Could not load the CSV file with any encoding")
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Find description and category columns
        desc_cols = [col for col in df.columns if 'description' in col.lower() or 'desc' in col.lower()]
        cat_cols = [col for col in df.columns if 'catégorie' in col.lower() or 'categorie' in col.lower() or 'category' in col.lower()]
        
        if not desc_cols:
            if 'Description' in df.columns:
                desc_cols = ['Description']
            elif 'Description.1' in df.columns:
                desc_cols = ['Description.1']
            elif 'Description.2' in df.columns:
                desc_cols = ['Description.2']
            else:
                print("Available columns:", df.columns.tolist())
                raise ValueError("Could not find description column")
        
        if not cat_cols:
            if 'Catégorie' in df.columns:
                cat_cols = ['Catégorie']
            elif 'Sous-catégorie' in df.columns:
                cat_cols = ['Sous-catégorie']
            else:
                print("Available columns:", df.columns.tolist())
                raise ValueError("Could not find category column")
        
        desc_col = desc_cols[0]
        cat_col = cat_cols[0]
        
        print(f"Using description column: {desc_col}")
        print(f"Using category column: {cat_col}")
        
        # Extract relevant columns
        df_clean = df[[desc_col, cat_col]].copy()
        df_clean.columns = ['description', 'category']
        
        # Remove rows with missing values
        df_clean = df_clean.dropna()
        
        # Analyze class distribution BEFORE filtering
        print("\n" + "="*50)
        print("ORIGINAL CLASS DISTRIBUTION ANALYSIS")
        print("="*50)
        category_counts = df_clean['category'].value_counts()
        print(category_counts)
        
        # Calculate class imbalance ratio
        max_count = category_counts.max()
        min_count = category_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 5:
            print("HIGH CLASS IMBALANCE DETECTED")
        
        # Filter out categories with too few examples
        min_examples = 99  # Reduced from 98 to keep more categories
        valid_categories = category_counts[category_counts >= min_examples].index
        df_clean = df_clean[df_clean['category'].isin(valid_categories)]
        
        print(f"\nData after filtering (>={min_examples} examples): {df_clean.shape}")
        print(f"Categories kept: {len(valid_categories)}")
        
        # Show category distribution after filtering
        print("\n" + "="*50)
        print("FINAL CLASS DISTRIBUTION")
        print("="*50)
        final_counts = df_clean['category'].value_counts()
        print(final_counts)
        
        # Show removed categories
        removed_categories = category_counts[category_counts < min_examples]
        if len(removed_categories) > 0:
            print(f"\nRemoved categories with <{min_examples} examples:")
            print(removed_categories)
        
        return df_clean
    
    def prepare_features(self, df):
        print("Cleaning and preparing text features...")
        
        # Clean text
        df['description_clean'] = df['description'].apply(self.clean_text)
        
        # Add domain features
        df['domain_features'] = df['description'].apply(self.add_domain_features)
        
        # Combine them
        df['enhanced_text'] = df['description_clean'] + ' ' + df['domain_features']
        
        # Remove empty descriptions
        df = df[df['description_clean'].str.len() > 0]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        df['category_encoded'] = self.label_encoder.fit_transform(df['category'])
        
        return df
    
    def train_model(self, df, test_size=0.2, model_type='logistic', balance_method='class_weight'):
        print("Training model with class balancing...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['enhanced_text'], 
            df['category_encoded'], 
            test_size=test_size, 
            random_state=42, 
            stratify=df['category_encoded']
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Analyze class distribution in training set
        print("\nTraining set class distribution:")
        train_dist = pd.Series(y_train).value_counts().sort_index()
        for idx, count in train_dist.items():
            category_name = self.label_encoder.inverse_transform([idx])[0]
            print(f"  {category_name}: {count} examples")
        
        # Create TF-IDF vectorizer with enhanced parameters
        self.vectorizer = TfidfVectorizer(
            max_features=8000,  # Increased from 5000
            ngram_range=(1, 3),  # Include trigrams
            min_df=1,  # Reduced to capture more features
            max_df=0.9,  # Slightly increased
            sublinear_tf=True  # Better handling of varying document lengths
        )
        
        # Transform text to features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Apply balancing technique
        if balance_method == 'smote':
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
        elif balance_method == 'undersample':
            print("Applying random undersampling...")
            undersampler = RandomUnderSampler(random_state=42)
            X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_tfidf, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_tfidf, y_train
        
        # Choose model with class balancing
        if model_type == 'logistic':
            if balance_method == 'class_weight':
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weight_dict = dict(zip(np.unique(y_train), class_weights))
                self.model = LogisticRegression(
                    random_state=42, 
                    max_iter=2000,
                    class_weight=class_weight_dict,
                    C=0.1  # Increased regularization
                )
            else:
                self.model = LogisticRegression(random_state=42, max_iter=2000, C=0.1)
                
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)  # Reduced smoothing
            
        elif model_type == 'random_forest':
            if balance_method == 'class_weight':
                self.model = RandomForestClassifier(
                    n_estimators=200, 
                    random_state=42, 
                    class_weight='balanced',
                    max_depth=10
                )
            else:
                self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        
        # Train model
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n" + "="*60)
        print(f"OVERALL MODEL ACCURACY: {accuracy:.2%}")
        print(f"="*60)
        
        # Detailed classification report
        target_names = self.label_encoder.classes_
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Show confusion matrix
        print("\nConfusion Matrix Analysis:")
        cm = confusion_matrix(y_test, y_pred)
        
        # Find most confused categories
        print("\nMost Common Misclassifications:")
        for i, true_cat in enumerate(target_names):
            for j, pred_cat in enumerate(target_names):
                if i != j and cm[i, j] > 0:
                    print(f"  {true_cat} → {pred_cat}: {cm[i, j]} cases")
        
        return X_test, y_test, y_pred
    
    def predict_with_details(self, text):
        """Enhanced prediction with detailed analysis"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet")
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Add domain features
        domain_features = self.add_domain_features(text)
        enhanced_text = cleaned_text + ' ' + domain_features
        
        # Transform to features
        text_tfidf = self.vectorizer.transform([enhanced_text])
        
        # Make prediction
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Get all categories with their probabilities
        categories = self.label_encoder.classes_
        prob_dict = dict(zip(categories, probabilities))
        
        # Sort by probability
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get top prediction
        top_category = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        return {
            'prediction': top_category,
            'confidence': confidence,
            'all_probabilities': sorted_probs,
            'cleaned_text': cleaned_text,
            'domain_features': domain_features,
            'enhanced_text': enhanced_text
        }
    
    def predict(self, text):
        """Simple prediction (backward compatibility)"""
        result = self.predict_with_details(text)
        return result['prediction'], result['confidence']
    
    def save_model(self, filename='enhanced_incident_classifier.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'stop_words': self.stop_words,
            'hardware_keywords': self.hardware_keywords,
            'application_keywords': self.application_keywords
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Enhanced model saved as {filename}")
    
    def load_model(self, filename='enhanced_incident_classifier.pkl'):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.stop_words = model_data['stop_words']
        self.hardware_keywords = model_data.get('hardware_keywords', self.hardware_keywords)
        self.application_keywords = model_data.get('application_keywords', self.application_keywords)
        
        print(f"Enhanced model loaded from {filename}")


def main():
    # File path
    file_path = r"C:\Users\laitouahmane\OneDrive - Balcan Innovations Inc\Desktop\ML classifier\incidents_2025-06-26_10-23-53.csv"
    
    # Initialize enhanced classifier
    classifier = EnhancedIncidentClassifier()
    
    try:
        df = classifier.load_and_preprocess_data(file_path)
        
        # Prepare features
        df = classifier.prepare_features(df)
        
        # Train model with different balancing techniques
        print("\n" + "="*60)
        print("TRAINING WITH CLASS BALANCING")
        print("="*60)
        
        # Try different balancing methods
        balance_methods = ['class_weight', 'smote']  # Can also try 'undersample'
        
        for method in balance_methods:
            print(f"\n Training with {method} balancing...")
            try:
                X_test, y_test, y_pred = classifier.train_model(
                    df, 
                    test_size=0.2, 
                    model_type='logistic',
                    balance_method=method
                )
                
                # Save model for this method
                classifier.save_model(f'enhanced_classifier_{method}.pkl')
                
                # Test with the problematic example
                print(f"\n Testing with '{method}' model:")
                result = classifier.predict_with_details("I need new headphones")
                print(f"Input: 'I need new headphones'")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Cleaned text: '{result['cleaned_text']}'")
                print(f"Domain features: '{result['domain_features']}'")
                print("\nTop 3 predictions:")
                for cat, prob in result['all_probabilities'][:3]:
                    print(f"  {cat}: {prob:.2%}")
                
                break  # Use the first successful method
                
            except Exception as e:
                print(f" Error with {method}: {e}")
                continue
        
        # Interactive testing with enhanced details
        print("\n" + "="*60)
        print("ENHANCED INTERACTIVE TESTING")
        print("="*60)
        print("Type 'quit' to exit, 'detail' for detailed analysis")
        
        while True:
            try:
                user_input = input("\nEnter incident description: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                    
                if not user_input:
                    print("Please enter a description")
                    continue
                
                if user_input.lower() == 'detail':
                    print("Detailed analysis mode - enter description:")
                    user_input = input("Description: ").strip()
                    if not user_input:
                        continue
                    
                    result = classifier.predict_with_details(user_input)
                    print(f"\n DETAILED ANALYSIS:")
                    print(f"Original text: '{user_input}'")
                    print(f"Cleaned text: '{result['cleaned_text']}'")
                    print(f"Domain features: '{result['domain_features']}'")
                    print(f"Enhanced text: '{result['enhanced_text']}'")
                    print(f"\n PREDICTION: {result['prediction']}")
                    print(f" CONFIDENCE: {result['confidence']:.2%}")
                    print(f"\n ALL PROBABILITIES:")
                    for cat, prob in result['all_probabilities']:
                        print(f"  {cat}: {prob:.2%}")
                else:
                    category, confidence = classifier.predict(user_input)
                    print(f"Predicted Category: {category}")
                    print(f"Confidence: {confidence:.2%}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your file path and data format")

if __name__ == "__main__":
    main()