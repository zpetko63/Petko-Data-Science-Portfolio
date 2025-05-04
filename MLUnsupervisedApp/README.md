# Unsupervised Machine Learning Final Project

## Project Overview

This project presents a Streamlit web application designed for exploring and analyzing structured datasets using unsupervised machine learning techniques. This app serves as a final project showcasing practical data science skills, including data loading, preprocessing, dimensionality reduction, and clustering.

The application allows users to work with either a built-in 2025 NFL Draft dataset or upload their own data. It provides an interactive interface to:
- Apply **Principal Component Analysis (PCA)** for dimensionality reduction and visualization.
- Perform **K-Means Clustering** and **Hierarchical Clustering** to identify inherent clusters in the data.
- Visualize clustering results in a 2D PCA space.
- Evaluate K-Means clustering using the Elbow Method and Silhouette Scores.
- Visualize Hierarchical Clustering using a Dendrogram.

This project contributes to my data science portfolio by demonstrating proficiency in building interactive data analysis tools and applying core unsupervised learning algorithms.

## Instructions

To run the Streamlit application (`MLUnsupervisedApp.py`), follow these steps:

1.  **Clone the Repository**

2.  **Install Dependencies:** Ensure you have the necessary Python libraries installed. Required libraries are listed in `requirements.txt`

4.  **Run the Streamlit App:** Navigate into the directory containing the Python script and run the app:
    ```bash
    cd MLUnsupervisedApp
    streamlit run MLUnsupervisedApp.py
    ```
    OR

    **Run in Streamlit Cloud:** [Link to Deployed App](https://petko-data-science-portfolio-74dezpd6ygwclv3ny2ifsk.streamlit.app/)

## Dataset Description

* **Default 2025 NFL Draft:** The primary default dataset, "BNB 2025 NFL Draft Database", is sourced from an Excel file from [BNB](https://bnbfootball.com/database/). It contains statistical data for various college football players eligible for the 2025 NFL Draft, broken down by position.
* **User Upload:** The app also supports user-uploaded datasets in `.csv` or `.xlsx` formats. Users are advised to provide clean, structured data with clear headers.

## Data Preprocessing Steps

The application performs the following preprocessing steps, depending on the selected dataset:

1.  **Data Loading:** Reads data from the selected source (default Excel file or user-uploaded CSV/Excel).
2.  **Position-Specific Handling (NFL Dataset):** Filters and combines data based on the user's selected NFL position group.
3.  **Initial Cleaning (NFL Dataset):** Handles specific formatting for columns like 'HT' (Height) and 'AGE' in the NFL dataset and removes rows with missing values (`-` or `NaN`).
4.  **Feature Selection (User Data):** Allows the user to select relevant numerical columns for analysis from their uploaded dataset.
5.  **Data Scaling:** Applies `StandardScaler` to the selected numerical features to normalize them, which is essential for distance-based algorithms like K-Means and Hierarchical Clustering.

## Model Implementation and Hyperparameter Selection

The app implements PCA and two clustering algorithms:

1.  **Principal Component Analysis (PCA):**
    * Used for dimensionality reduction to 2 principal components. This specific hyperparameter is fixed within the code to allow for 2D visualization.
    * The principal components capture the most variance in the scaled data.

2.  **K-Means Clustering:**
    * A centroid-based clustering algorithm.
    * **Hyperparameter (`k` - number of clusters):** This is selected interactively by the user via a **Streamlit slider**, allowing exploration of different cluster numbers (currently ranges from 2 to 5).
    * **Evaluation:** The app provides plots for the **Elbow Method** (WCSS vs k) and **Silhouette Scores** (Silhouette Score vs k) to help the user evaluate the quality of clustering for different `k` values and inform the selection of an appropriate number of clusters.

3.  **Hierarchical Clustering (Agglomerative):**
    * Builds a hierarchy of clusters. The implementation uses `AgglomerativeClustering`.
    * **Linkage Method:** The code uses the 'ward' linkage method (`linkage='ward'`), which minimizes the variance of the clusters being merged. This is a fixed choice in the implementation.
    * **Hyperparameter (`k` - number of clusters):** This is selected interactively by the user via a **Streamlit slider** (currently ranges from 2 to 5).
    * **Visualization:** A **dendrogram** is generated, visually representing the hierarchy of clusters and aiding the user in deciding the number of clusters by inspecting the tree structure.

## Visualizations and Outputs

The application generates several interactive visualizations using Plotly:

* **2D PCA Projection:** A scatter plot showing data points in the reduced 2D PCA space. When clustering is applied, points are colored by their assigned cluster. Hovering over points in the NFL dataset shows player details.

![2D PCA Plot](`PCA 2d Projection.png`)

* **K-Means Evaluation Plots:**
    * **Elbow Method Plot:** Shows the Within-Cluster Sum of Squares (WCSS) for different `k` values to help identify a potential "elbow" point.
    * **Silhouette Score Plot:** Shows the silhouette score for different `k` values, providing a measure of how well each data point fits within its assigned cluster compared to other clusters.
    * *(Placeholder: `![KMeans Evaluation Plots](images/kmeans_evaluation.png)`)*
* **Hierarchical Clustering Dendrogram:** A tree-like diagram illustrating the arrangement of clusters produced by hierarchical clustering.
    * *(Placeholder: `![Hierarchical Clustering Dendrogram](images/dendrogram.png)`)*
* **Clustering Results Table (Toggleable):** Displays the dataset with added columns for the PCA components (PC1, PC2) and the assigned cluster labels for K-Means or Hierarchical Clustering, viewable via a toggle switch.

## References

* **Insert Links Here:**
    * Links to the official documentation for libraries used (e.g., Streamlit, pandas, scikit-learn, plotly, scipy).
    * Links to any specific tutorials, articles, or guides that informed your approach to PCA, K-Means, Hierarchical Clustering, or their evaluation/visualization.
    * Link to the original source of the NFL dataset (if publicly available online, beyond just the raw GitHub file link).

## Visual Examples

*(Include screenshots of your running application here to illustrate its features and the generated visualizations. You can add images by placing them in a folder (e.g., `images`) in your repository and linking them like this: `![Alt text describing the image](images/your_screenshot_name.png)`)*.

* Screenshot showing the dataset selection and upload feature.
* Screenshot showing the PCA visualization.
* Screenshot showing the K-Means section with the scatter plot and evaluation plots.
* Screenshot showing the Hierarchical Clustering section with the dendrogram and scatter plot.

---

**Project Author:** Zach Petko
