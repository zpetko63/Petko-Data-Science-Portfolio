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
1. **Access App in Streamlit Cloud**: [Link to app in Streamlit Cloud](https://petko-data-science-portfolio-bpybmuqcd3dcukxw9rnqg5.streamlit.app/)
1. **Choose a Dataset**: Select the Breast Cancer dataset or upload your own CSV file.
2. **Select Features**: Choose the columns for the features (X) and the target (Y) in the dataset.
3. **Choose a Model**: Select either Logistic Regression or Decision Tree.
4. **Adjust Hyperparameters**:
   - For Logistic Regression, adjust the maximum number of iterations.
   - For Decision Tree, adjust the maximum depth and minimum samples required to split.
5. **View Model Evaluation**: After training the model, view metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC curve. If a Decision Tree is chosen, view the tree visualization.
6. **Interact with the App**: Adjust features, models, and hyperparameters to explore how they affect the results.


#####################################################################################3

# Machine Learning Application Project

## Project Overview

This project is a Streamlit app that allows users to explore machine learning models on a dataset of their choice. Users can upload their own CSV data or use the loaded Breast Cancer dataset. The app provides an interface for selecting features, training models, and evaluating their performance. Users can experiment with Logistic Regression and Decision Tree models as well as adjust hyperparameters to fine-tune the model.

## Instructions

To run the Streamlit application (`ML Streamlit App.py`), follow these steps:

1.  **Clone the Repository**
  
2.  **Install Dependencies:** Ensure you have the necessary Python libraries installed. Use the `requirements.txt` file to ensure the correct versions are installed

3.  **Run the Streamlit App:** Navigate into the directory containing the Python script and run the app:
    ```bash
    cd MLApplicationApp
    streamlit run MLApplicationApp.py
    ```
    OR
    
 **Run in Streamlit Cloud:** [Link to streamlit cloud](https://petko-data-science-portfolio-bpybmuqcd3dcukxw9rnqg5.streamlit.app/)

## Dataset Description

* **Default Dataset (Breast Cancer):** The app includes the built-in Breast Cancer dataset from scikit-learn. This dataset is commonly used for binary classification tasks.
* **User Upload:** The app supports user-uploaded datasets in `.csv` or `.xlsx` formats. Users are advised to provide clean, structured data with clear headers and a column suitable for use as the target variable (Y).

## Data Preprocessing Steps

The application performs the following preprocessing steps, depending on the selected dataset:

1.  **Data Loading:** Reads data from the selected source (default Breast Cancer dataset or user-uploaded CSV/Excel file).
2.  **Feature Selection:** Allows the user to interactively select the predictor features (X columns) and the target variable (Y column) from the loaded dataset using sidebar selectboxes.
3.  **Data Splitting:** Splits the selected features (X) and target variable (Y) into training and testing sets using `train_test_split` with a test size of 20%.

## Model Implementation and Hyperparameter Selection

The app implements two supervised machine learning models for binary classification:

1.  **Logistic Regression:**
    * A linear model for binary classification.
    * **Hyperparameter (`max_iter` - Maximum Iterations):** This is selected interactively allowing tuning of the maximum number of iterations for the solver to converge.

  
      
2.  **Decision Tree:**
    * **Hyperparameters (`max_depth`, `min_samples_split`):** These are selected interactively by the user via **Streamlit sliders**, allowing tuning of the maximum depth of the tree (1 to 20) and the minimum number of samples required to split an internal node (2 to 20).

After selecting the model and tuning hyperparameters, the model is trained on the training data (`X_train`, `Y_train`) and used to make predictions (`Y_pred`) and predict probabilities (`Y_prob`) on the test data (`X_test`).

## Visualizations and Outputs

The application provides several key outputs and visualizations for model evaluation:

* **Selected Dataset View:** Displays a dataframe showing only the selected X and Y features.
* **Evaluation Metrics:** Displays common classification metrics calculated using scikit-learn:
    * Accuracy Score
    * Precision Score (weighted)
    * Recall Score (weighted)
    * F1-Score (weighted)
    * Confusion Matrix
* **ROC Curve:** Plots the Receiver Operating Characteristic (ROC) curve, which illustrates the model's performance at different classification thresholds, along with the Area Under the Curve (AUC) score.
    * *(Placeholder: `![ROC Curve](images/roc_curve.png)`)*
* **Decision Tree Visualization:** If the Decision Tree model is selected, the app displays a plot of the trained decision tree structure, showing how the model makes decisions based on the selected features.
    * *(Placeholder: `![Decision Tree Plot](images/decision_tree.png)`)*

## Visual Examples

*(Include screenshots of your running application here to illustrate its features and the generated plots. You can add images by placing them in a folder (e.g., `images`) in your repository and linking them like this: `![Alt text describing the image](images/your_screenshot_name.png)`)*.

* Screenshot showing the sidebar controls and dataset/feature selection.
* Screenshot showing the selected dataset display.
* Screenshot showing the evaluation metrics section.
* Screenshot showing the ROC Curve.
* Screenshot showing the Decision Tree visualization.

---

**Project Author:** Zach Petko

