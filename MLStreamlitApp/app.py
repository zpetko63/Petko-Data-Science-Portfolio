#######################################################
### Machine Learning Application Project
### Zach Petko
#######################################################

# Description:
#   This streamlit python file creates a streamlit app that can apply various 
#   machine learning models to a user uploaded dataset or for a sample dataset
#   with the ability to train, tune, and evaluate the model.

# Packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_breast_cancer

# App Structure
st.title('Machine Learning Application Project')
st.write('To start, use the sidebar to select your data then a machine learning model to use.')

# Dataset Selection
st.sidebar.subheader("1. Choose a Dataset")
select_df = st.sidebar.selectbox('Select Dataset', ('Breast Cancer', 'User Input'))

df = None
if select_df == "Breast Cancer":
    data = load_breast_cancer(as_frame=True)
    df = data.frame
else:
    uploaded_file = st.sidebar.file_uploader('Input your CSV data here', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Feature Selection
if df is not None:
    st.sidebar.subheader("2. Select Features")
    cols = df.columns.tolist()

    # Set defaults for breast cancer dataset
    default_x = ['mean radius'] if 'mean radius' in cols else cols[:1]
    default_y = 'target' if 'target' in cols else cols[-1]

    x_cols = st.sidebar.multiselect("Select X Features", options=cols, default=default_x)
    y_options = [c for c in cols if c not in x_cols]
    default_y_index = y_options.index('target') if 'target' in y_options else 0
    y_col = st.sidebar.selectbox("Select Y Feature", options=y_options, index=default_y_index)

    if x_cols and y_col:
        X = df[x_cols]
        Y = df[y_col]
        df_show = df[x_cols + [y_col]]

        st.subheader(f"{select_df} Dataset with selected features")
        st.write(f"Selected X columns: {x_cols}")
        st.write(f"Selected Y column: {[y_col]}")
        st.dataframe(df_show)

        # Model Selection
        st.sidebar.subheader("3. Choose a Machine Learning Model")
        select_model = st.sidebar.selectbox('Select Machine Learning Model', ('Logistic Regression', 'Decision Tree'))

        # Split Data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        if select_model == "Logistic Regression":
            max_iter = st.sidebar.slider('Maximum Iterations', min_value=100, max_value=500, value=200, step=10)
            model = LogisticRegression(max_iter=max_iter)
        elif select_model == "Decision Tree":
            max_depth = st.sidebar.slider('Max Depth', 1, 20, 5)
            min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        Y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluation Metrics
        st.subheader(f"{select_model} Model Evaluation")

        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='weighted')
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        conf_matrix = confusion_matrix(Y_test, Y_pred)
        fpr, tpr, _ = roc_curve(Y_test, Y_prob)
        roc_auc = auc(fpr, tpr)

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        # ROC Curve
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], 'k--', label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        if select_model == "Decision Tree":
            st.subheader("Decision Tree Visualization")
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(model, filled=True, feature_names=x_cols, class_names=df[y_col].astype(str), rounded=True, ax=ax)
            st.pyplot(fig)
    else:
        st.subheader("Please select both X and Y features.")
else:
    st.subheader("Upload a dataset or select one from the sidebar to begin.")

# To run streamlit app run terminal command: streamlit run (filepath)
