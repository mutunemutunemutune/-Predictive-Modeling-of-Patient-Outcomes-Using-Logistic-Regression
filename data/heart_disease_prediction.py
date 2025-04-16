import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
import os
import requests
from io import StringIO
warnings.filterwarnings('ignore')

def download_dataset():
    """Download the Heart Disease UCI dataset if it doesn't exist."""
    if not os.path.exists('heart.csv'):
        print("Downloading Heart Disease UCI dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Add column names to the data
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                      'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            
            # Read the data and add column names
            df = pd.read_csv(StringIO(response.text), names=columns, na_values='?')
            
            # Convert columns to appropriate data types
            numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert categorical columns to integers
            categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
            for col in categorical_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Save to CSV
            df.to_csv('heart.csv', index=False)
            print("Dataset downloaded and saved as heart.csv")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return False
    return True

# Load the dataset
def load_data():
    # Try to download the dataset if it doesn't exist
    if not download_dataset():
        return None
        
    try:
        # Load the data
        df = pd.read_csv('heart.csv')
        
        # Convert target to binary (0 or 1)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Data preprocessing
def preprocess_data(df):
    if df is None:
        return None
        
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Handle missing values (if any)
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical variables to dummy variables
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    for col in categorical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mode()[0])
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Feature scaling for numerical features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

# Train the model
def train_model(X_train, y_train):
    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    if model is None:
        return
        
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def main():
    # Load and preprocess data
    df = load_data()
    if df is None:
        return
        
    df = preprocess_data(df)
    if df is None:
        return
    
    # Split the data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    if model is None:
        return
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_[0]
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nFeature Importance Analysis:")
    print(feature_importance)

if __name__ == "__main__":
    main() 