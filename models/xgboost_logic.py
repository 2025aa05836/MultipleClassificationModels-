import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import io

def run_xgboost_evaluation(X, y, display_labels_arr):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate and train the model
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') # Suppress warning
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Check if there are at least two classes in y_test for roc_auc_score
    if len(y_test.unique()) > 1:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
    else:
        auc_score = 0.0

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Generate Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels_arr)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix - XGBoost Classifier')

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
