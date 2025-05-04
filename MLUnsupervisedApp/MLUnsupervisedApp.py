#######################################################
### Unsupervised Machine Learning Final Project
### Zach Petko
#######################################################

# Packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import plotly.figure_factory as ff


#######################################################
# 0 - App Introduction
st.title('Unsupervised Machine Learning Final Project')
st.write('This Streamlit web application serves as the final project for a ' \
'Data Science course. Designed and implemented by Zach Petko, the tool enables ' \
'users to explore and analyze structured datasets—either a built-in 2025 NFL Draft ' \
'dataset or user-uploaded data—using unsupervised learning techniques. The app ' \
'provides an interactive interface to perform dimensionality reduction via Principal ' \
'Component Analysis (PCA), as well as clustering through K-Means and Hierarchical ' \
'Clustering. Users can select specific features, visualize clusters in 2D PCA space, '
'and evaluate clustering performance using metrics like the silhouette score. The ' \
'project showcases the power of unsupervised learning in identifying structure and ' \
'patterns within complex datasets, with a particular focus on interpretability and ' \
'interactive data exploration. Follow the directions below to use the features in' \
'this unsupervised machine learning app. Be patient when using the app as it may take ' \
'a moment to load.')
st.write('---')

#######################################################
# 1 - Dataset Selection
st.write('### 1. Choose to use the 2025 NFL Draft Data or Input your own data')
st.write('Choose a dataset to work with. You can use the default 2025 NFL Draft ' \
'dataset or upload your own CSV file.')

#   Radio Selection
df_selection = st.radio('Select Dataset',{'Input My Own Dataset','2025 NFL Draft Dataset'})

#   Load 2025 NFL Draft Dataset

#       Position Selection Dict
position_dict = {
    "Quarterback": "QB",
    "Wide Receiver": "WR",
    "Running Back": "RB",
    "Fullback": "FB",
    "Tight End": "TE",
    "Offensive Line (OT/G/C)": "OT_G_C",
    "Defensive Tackle / Edge Rusher": "DT_EDGE",
    "Linebacker": "LB",
    "Safety": "S",
    "Cornerback": "CB",
    "Kicker": "K",
    "Punter": "P"}

#       Position Column Dict
column_dict = {
    "Quarterback": 18,     # S
    "Wide Receiver": 18,   # S
    "Running Back": 15,    # P
    "Fullback": 18,        # S
    "Tight End": 17,       # R
    "Linebacker": 16,      # Q
    "Safety": 15,          # P
    "Cornerback": 17,      # R
    "Kicker": 16,          # P
    "Punter": 17           # Q
    }

if df_selection == '2025 NFL Draft Dataset':
    #   Position Group Selection
    position_selection = st.selectbox('Select Position Group', position_dict)
    
    ##  OL Position Selection 
    if position_selection == 'Offensive Line (OT/G/C)':
        temp1 = pd.read_excel('https://raw.githubusercontent.com/zpetko63/Petko-Data-Science-Portfolio/main/MLUnsupervisedApp/BNB_2025_NFL_Draft.xlsx'
,sheet_name='OT')[['NAME','COLLEGE','C-POS','AGE','HT','WT','GMS','YRS','SACK','HIT','HURRY','LOSS','PEN']]
        temp2 = pd.read_excel('https://raw.githubusercontent.com/zpetko63/Petko-Data-Science-Portfolio/main/MLUnsupervisedApp/BNB_2025_NFL_Draft.xlsx'
,sheet_name='G')[['NAME','COLLEGE','C-POS','AGE','HT','WT','GMS','YRS','SACK','HIT','HURRY','LOSS','PEN']]
        temp3 = pd.read_excel('https://raw.githubusercontent.com/zpetko63/Petko-Data-Science-Portfolio/main/MLUnsupervisedApp/BNB_2025_NFL_Draft.xlsx'
,sheet_name='C')[['NAME','COLLEGE','C-POS','AGE','HT','WT','GMS','YRS','SACK','HIT','HURRY','LOSS','PEN']]
        fb_df = pd.concat([temp1,temp2,temp3],ignore_index=True)
        fb_df = fb_df.rename(columns={'C-POS':'POS'})

    ##  DL/EDGE Position Selection
    elif position_selection == 'Defensive Tackle / Edge Rusher':
        temp1 = pd.read_excel('https://raw.githubusercontent.com/zpetko63/Petko-Data-Science-Portfolio/main/MLUnsupervisedApp/BNB_2025_NFL_Draft.xlsx'
,sheet_name='DT')[['NAME','COLLEGE','AGE','HT','WT','POS','GMS','TCK','TFL','SCK','PD','FF','PRES%','MISS%']]
        temp2 = pd.read_excel('https://raw.githubusercontent.com/zpetko63/Petko-Data-Science-Portfolio/main/MLUnsupervisedApp/BNB_2025_NFL_Draft.xlsx'
,sheet_name='EDGE')[['NAME','COLLEGE','AGE','HT','WT','POS','GMS','TCK','TFL','SCK','PD','FF','PRES%','MISS%']]
        fb_df = pd.concat([temp1,temp2],ignore_index=True)

    ##  All Other Positions
    else:
        fb_df = pd.read_excel('https://raw.githubusercontent.com/zpetko63/Petko-Data-Science-Portfolio/main/MLUnsupervisedApp/BNB_2025_NFL_Draft.xlsx'
,sheet_name=position_dict[position_selection]).iloc[:,3:column_dict[position_selection]]

    #   Position DF Cleaning

    ##  Kicker and Punter Column Clean
    if position_selection in ['Kicker', 'Punter']:
        fb_df = fb_df.drop('KO', axis=1) # Drop kickoff column

    ##  Drop Missing Values
    fb_df = fb_df.replace('-', np.nan) # Replace - with na
    fb_df = fb_df.dropna()

    ##  Format Height
    fb_df[['Ft','In']] = fb_df['HT'].str.extract(r"(\d+)'(\d+)")
    fb_df['HT'] = fb_df['Ft'].astype(int)*12 + fb_df['In'].astype(int)
    fb_df = fb_df.drop(['Ft', 'In'], axis=1)

    ##  Format Age
    fb_df['AGE'] = fb_df['AGE'].str.extract(r"(\d+)").astype(int)

    df = fb_df


#   Load User Inputted Dataset
elif df_selection == 'Input My Own Dataset':
    st.markdown("""
Please upload a `.csv` or `.xlsx` file with a clear structure. Your dataset should:
- Include column headers in the first row
- Have no completely empty columns or rows
- Use consistent formatting (numeric columns should not include symbols like `$`, `%`, or `-`)
- Avoid merged cells or multi-level headers
- Handle missing values as empty cells or `NaN`, not special characters

Failure to follow these directions will likely lead to an error.
""")

    df = st.file_uploader('Upload your Dataset Here', type=['csv','xlxs'])

#   Toggle Show Dataset
dataset_show_toggle = st.toggle('Show Dataset')
if dataset_show_toggle:
    st.dataframe(df)

st.write('---')

#######################################################
# 2 - Unsupervised ML Model Selection
st.write('### 2. Select an Unsupervised Machine Learning Model to use')
st.write('Select the **Unsupervised Machine Learning** model that you would like to' \
         'use for dimentionality reduction or clustering.')

#   Radio Selection
model_selection = st.radio('Select Unsupervised Machine Learning Model',
                           {'Principal Component Analysis (PCA)',
                            'Hierarchical Clustering',
                            'K-Means'})


st.write('---')
#######################################################
# 3 - Model Implementation
st.write(f'### 3. {model_selection} Implementation')

#   Seperate Data
if df_selection == '2025 NFL Draft Dataset':
    if position_selection in ['Fullback','Offensive Line (OT/G/C)','Defensive Tackle / Edge Rusher']:
        X = df.drop(['NAME', 'COLLEGE','POS'], axis=1)
    else:
        X = df.drop(['NAME', 'COLLEGE'], axis=1)
elif df_selection == 'Input My Own Dataset':
    if df:
        x_cols = st.multiselect("Select features (columns) for X:", df.columns)
        X = df[x_cols]
    else:
        st.warning('Please upload a dataset first!')
        st.stop()


#   Scale Data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
    
#   Principal Component Analysis (PCA)
if model_selection == 'Principal Component Analysis (PCA)':
    st.write('This section applies PCA to reduce your selected features into two ' \
        'principal components for visualization. The scatter plot displays the data ' \
        'in 2D PCA space, helping you visually detect natural groupings or patterns.')

    #   PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1]})
    
    #   Create dataframe with k means and names added back
    df_with_pca = df
    df_with_pca['PC1'] = pca_df['PC1']
    df_with_pca['PC2'] = pca_df['PC2']
    
    #   k means plotly scatterplot using pca
    fig = px.scatter(
        df_with_pca,
        x='PC1',
        y='PC2',
        title='2D PCA Projection',
        labels={'PC1': 'Principal Component 1','PC2': 'Principal Component 2'}) 

    if df_selection == '2025 NFL Draft Dataset':
        if position_selection in ['Fullback','Offensive Line (OT/G/C)','Defensive Tackle / Edge Rusher']:
            fig.update_traces(
            hovertemplate="<br>".join([
                'Name: %{customdata[0]}',
                'College: %{customdata[1]}',
                'Position: %{customdata[2]}',
                'PC 1: %{x}',
                'PC 2: %{y}']),
            customdata=df_with_pca[['NAME','COLLEGE','POS']].values)
        else:
            fig.update_traces(
            hovertemplate="<br>".join([
                'Name: %{customdata[0]}',
                'College: %{customdata[1]}',
                'PC 1: %{x}',
                'PC 2: %{y}']),
            customdata=df_with_pca[['NAME','COLLEGE']].values) 
    
    st.plotly_chart(fig, use_container_width=True)

    #   Toggle Show pca Dataset
    pca_dataset_show_toggle = st.toggle('Show PCA Dataset')
    if pca_dataset_show_toggle:
        st.dataframe(df_with_pca)





#   Hierarchical Clustering
elif model_selection == 'Hierarchical Clustering':
    st.write('Select the number of clusters to create using Agglomerative (hierarchical) ' \
    'clustering. This method forms clusters based on a hierarchy of merged points. ' \
    'Results are displayed similarly to K-Means, allowing for direct comparison.')
    
    #   Plot dendrogram
    Z = linkage(X_std, method='ward')
    fig = ff.create_dendrogram(X_std, linkagefun=lambda x: linkage(x, method='ward'))
    fig.update_layout(title='Hierarchical Clustering Dendrogram')
    st.plotly_chart(fig, use_container_width=True)  

    #   Select k clusters
    k = st.slider('Select k value for number of clusters', 2,5)
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    df['Cluster'] = agg.fit_predict(X_std)
    st.write('\nCluster sizes:\n', df['Cluster'].value_counts())

    #   Plot clusters with pca

    ##   PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': df["Cluster"]})
    
    #   Create dataframe with k means and names added back
    df_with_pca = df
    df_with_pca['PC1'] = pca_df['PC1']
    df_with_pca['PC2'] = pca_df['PC2']
    df_with_pca['Cluster'] = pca_df['Cluster'].astype(str)
    
    #   k means plotly scatterplot using pca
    fig = px.scatter(
        df_with_pca,
        x='PC1',
        y='PC2',
        color='Cluster',
        title='Hierarchical Clustering: 2D PCA Projection',
        labels={'PC1': 'Principal Component 1','PC2': 'Principal Component 2'}) 

    if df_selection == '2025 NFL Draft Dataset':
        if position_selection in ['Fullback','Offensive Line (OT/G/C)','Defensive Tackle / Edge Rusher']:
            fig.update_traces(
            hovertemplate="<br>".join([
                'Name: %{customdata[0]}',
                'College: %{customdata[1]}',
                'Position: %{customdata[2]}',
                'PC 1: %{x}',
                'PC 2: %{y}',
                'Cluster: %{customdata[3]}']),
            customdata=df_with_pca[['NAME','COLLEGE','POS','Cluster']].values)
        else:
            fig.update_traces(
            hovertemplate="<br>".join([
                'Name: %{customdata[0]}',
                'College: %{customdata[1]}',
                'PC 1: %{x}',
                'PC 2: %{y}',
                'Cluster: %{customdata[2]}']),
            customdata=df_with_pca[['NAME','COLLEGE','Cluster']].values)
    
    st.plotly_chart(fig, use_container_width=True)

    #   Toggle Show pca Dataset
    HC_dataset_show_toggle = st.toggle('Show Hierarchical Clustering Dataset')
    if HC_dataset_show_toggle:
        st.dataframe(df_with_pca)


#   K-Means
elif model_selection == 'K-Means':
    st.write('Use the slider to choose the number of clusters you want K-Means to ' \
    'form. The data will be clustered and visualized in PCA space. The silhouette ' \
    'score is also provided to evaluate the quality of the clustering—higher is ' \
    'generally better.')

    #   Perform k means
    k = st.slider('Select k value for number of clusters', 2,5)
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(X_std)

    #   PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters})
    
    #   Create dataframe with k means and names added back
    df_with_pca = df
    df_with_pca['PC1'] = pca_df['PC1']
    df_with_pca['PC2'] = pca_df['PC2']
    df_with_pca['Cluster'] = pca_df['Cluster'].astype(str)
    
    #   k means plotly scatterplot using pca
    fig = px.scatter(
        df_with_pca,
        x='PC1',
        y='PC2',
        color='Cluster',
        title='KMeans Clustering: 2D PCA Projection',
        labels={'PC1': 'Principal Component 1','PC2': 'Principal Component 2'}) 

    if df_selection == '2025 NFL Draft Dataset':
        if position_selection in ['Fullback','Offensive Line (OT/G/C)','Defensive Tackle / Edge Rusher']:
            fig.update_traces(
            hovertemplate="<br>".join([
                'Name: %{customdata[0]}',
                'College: %{customdata[1]}',
                'Position: %{customdata[2]}',
                'PC 1: %{x}',
                'PC 2: %{y}',
                'Cluster: %{customdata[3]}']),
            customdata=df_with_pca[['NAME','COLLEGE','POS','Cluster']].values)
        else:
            fig.update_traces(
            hovertemplate="<br>".join([
                'Name: %{customdata[0]}',
                'College: %{customdata[1]}',
                'PC 1: %{x}',
                'PC 2: %{y}',
                'Cluster: %{customdata[2]}']),
            customdata=df_with_pca[['NAME','COLLEGE','Cluster']].values)
    
    st.plotly_chart(fig, use_container_width=True)

    #   Silhouette Score
    ks = [2,3,4,5] 
    wcss = []               
    silhouette_scores = []  
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_std)
        wcss.append(km.inertia_) 
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Method for Optimal k", "Silhouette Score for Optimal k"))

    ##  Elbow plot
    fig.add_trace(
        go.Scatter(x=ks, y=wcss, mode='lines+markers', name='WCSS'),
        row=1, col=1
    )

    ##  Silhouette plot
    fig.add_trace(
        go.Scatter(x=ks, y=silhouette_scores, mode='lines+markers', name='Silhouette Score', line=dict(color='green')),
        row=1, col=2
    )

    ##  Format Plots
    fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=1)
    fig.update_yaxes(title_text="Within-Cluster Sum of Squares", row=1, col=1)
    fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_layout(title_text="Clustering Evaluation Metrics", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    #   Toggle Show pca Dataset
    KM_dataset_show_toggle = st.toggle('Show K Means Dataset')
    if KM_dataset_show_toggle:
        st.dataframe(df_with_pca)


st.write('---')