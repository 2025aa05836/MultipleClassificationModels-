import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io

# Import model logic functions
from models.logistic_regression_logic import run_logistic_regression_evaluation
from models.decision_tree_logic import run_decision_tree_evaluation
from models.knn_logic import run_knn_evaluation
from models.naive_bayes_logic import run_naive_bayes_evaluation
from models.random_forest_logic import run_random_forest_evaluation
from models.xgboost_logic import run_xgboost_evaluation

# Set the title of the Streamlit application
st.title('Machine Learning Model Evaluation Dashboard')

# Define the list of available models
available_models = [
    'Logistic Regression',
    'Decision Tree Classifier',
    'K-Nearest Neighbor Classifier',
    'Naive Bayes Classifier',
    'Random Forest',
    'XGBoost'
]

# Add a sidebar for model selection
st.sidebar.title('Model Selection')
selected_model = st.sidebar.selectbox(
    'Choose a model:',
    available_models
)

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded and read into DataFrame!")

        # Display the first few rows of the DataFrame
        st.subheader("Preview of your data:")
        st.dataframe(df.head())

        st.subheader("Data Preprocessing and Model Training")

        # Automatically identify the last column as the target variable
        all_columns = df.columns.tolist()
        target_column = all_columns[-1] # Last column is the target
        st.write(f"Automatically selected target variable: **{target_column}** (last column)")

        # Separate features (X) and target (y)
        X = df.iloc[:, :-1] # All columns except the last one are features
        y = df[target_column] # The last column is the target

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns

        # One-hot encode categorical features
        if not categorical_cols.empty:
            X_categorical = pd.get_dummies(X[categorical_cols], drop_first=True)
        else:
            X_categorical = pd.DataFrame(index=X.index) # Empty DataFrame if no categorical cols

        # Scale numerical features
        X_numerical = X[numerical_cols]
        if not numerical_cols.empty:
            scaler = StandardScaler()
            X_numerical_scaled = scaler.fit_transform(X_numerical)
            X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_cols, index=X.index)
        else:
            X_numerical_scaled = pd.DataFrame(index=X.index) # Empty DataFrame if no numerical cols

        # Concatenate preprocessed features
        X_processed = pd.concat([X_numerical_scaled, X_categorical], axis=1)

        st.write(f"Selected model: **{selected_model}**")

        results = None
        if selected_model == 'Logistic Regression':
            results = run_logistic_regression_evaluation(X_processed, y)
        elif selected_model == 'Decision Tree Classifier':
            results = run_decision_tree_evaluation(X_processed, y)
        elif selected_model == 'K-Nearest Neighbor Classifier':
            results = run_knn_evaluation(X_processed, y)
        elif selected_model == 'Naive Bayes Classifier':
            results = run_naive_bayes_evaluation(X_processed, y)
        elif selected_model == 'Random Forest':
            results = run_random_forest_evaluation(X_processed, y)
        elif selected_model == 'XGBoost':
            results = run_xgboost_evaluation(X_processed, y)

        if results:
            st.subheader(f"Evaluation Metrics for {selected_model}")
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
            with col2:
                st.metric("AUC Score", f"{results['auc_score']:.4f}")
            with col3:
                st.metric("Precision", f"{results['precision']:.4f}")
            with col4:
                st.metric("Recall", f"{results['recall']:.4f}")
            with col5:
                st.metric("F1 Score", f"{results['f1_score']:.4f}")
            with col6:
                st.metric("MCC Score", f"{results['mcc_score']:.4f}")

            st.subheader(f"Confusion Matrix for {selected_model}")
            st.image(results['confusion_matrix_plot'], use_column_width=True)

    except Exception as e:
        st.error(f"Error processing file or running model: {e}")
else:
    st.info("Please upload a CSV file to get started.")
