import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import io

def run_logistic_regression_evaluation(X, y, display_labels_arr):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate and train the model
    model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' solver works well for small datasets and binary classification
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Generate Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels_arr)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix - Logistic Regression')

    # Save plot to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig) # Close the plot to prevent it from displaying twice in Streamlit
    buf.seek(0)

    return {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc_score': mcc,
        'confusion_matrix_plot': buf
    }
