# Machine Learning Application Project

## Overview
This project is a Streamlit app that allows users to explore machine learning models on a dataset of their choice. Users can upload their own CSV data or use the loaded Breast Cancer dataset. The app provides an interface for selecting features, training models, and evaluating their performance. Users can experiment with Logistic Regression and Decision Tree models as well as adjust hyperparameters to fine-tune the model.

The app provides evaluation metrics including accuracy, precision, recall, F1-score, and AUC. Additionally, the app generates confusion matrices and ROC curves to visually assess model performance. For Decision Trees, a visualization of the tree structure is provided.

## App Structure
- **Dataset Selection**: Users can choose between the Breast Cancer dataset or upload their own binary classification CSV dataset.
- **Feature Selection**: Users can select which columns to use as features (X) and the target (Y).
- **Model Selection**: Users can select from two machine learning models: Logistic Regression and Decision Tree.
- **Model Hyperparameter Tuning**: For both models, users can adjust hyperparameters such as maximum iterations (Logistic Regression) or maximum depth and minimum samples split (Decision Tree).
- **Model Evaluation**: Once the model is trained, performance is evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - ROC Curve and AUC
- **Decision Tree Visualization**: If a Decision Tree is chosen, the app will visualize the structure of the tree.

## How to Use
1. **Access App in Streamlit Cloud**: [Link to app in Streamlit Cloud] (https://petko-data-science-portfolio-bpybmuqcd3dcukxw9rnqg5.streamlit.app/)
1. **Choose a Dataset**: Select the Breast Cancer dataset or upload your own CSV file.
2. **Select Features**: Choose the columns for the features (X) and the target (Y) in the dataset.
3. **Choose a Model**: Select either Logistic Regression or Decision Tree.
4. **Adjust Hyperparameters**:
   - For Logistic Regression, adjust the maximum number of iterations.
   - For Decision Tree, adjust the maximum depth and minimum samples required to split.
5. **View Model Evaluation**: After training the model, view metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC curve. If a Decision Tree is chosen, view the tree visualization.
6. **Interact with the App**: Adjust features, models, and hyperparameters to explore how they affect the results.

