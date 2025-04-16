# Heart Disease Prediction using Logistic Regression

This project implements a logistic regression model to predict the likelihood of heart disease in patients based on various medical attributes.

## Project Overview

The project uses the Heart Disease UCI dataset from Kaggle to build a classification model that can assist in early detection of heart disease. The model takes into account various medical attributes such as age, sex, chest pain type, blood pressure, cholesterol levels, and other relevant factors.

## Features

- Data preprocessing and cleaning
- Feature scaling using StandardScaler
- Logistic Regression model implementation
- Comprehensive model evaluation metrics
- Visualization of results including:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance Plot

## Requirements

The project requires the following Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

Install the requirements using:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is the Heart Disease UCI dataset, which contains the following attributes:
- age
- sex
- cp (chest pain type)
- trestbps (resting blood pressure)
- chol (serum cholesterol)
- fbs (fasting blood sugar)
- restecg (resting electrocardiographic results)
- thalach (maximum heart rate achieved)
- exang (exercise-induced angina)
- oldpeak (ST depression induced by exercise)
- slope (slope of the peak exercise ST segment)
- ca (number of major vessels)
- thal (thalassemia)
- target (1 = disease, 0 = no disease)

## Usage

1. Place the heart.csv dataset in the project directory
2. Run the main script:
```bash
python heart_disease_prediction.py
```

The script will:
- Load and preprocess the data
- Train the logistic regression model
- Evaluate the model performance
- Generate visualization plots

## Output

The script generates three visualization files:
- confusion_matrix.png
- roc_curve.png
- feature_importance.png

It also prints the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## License

This project is open source and available under the MIT License. 