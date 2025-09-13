import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import scipy
outputPath = 'd://z/master/comvis/color_indices/outputs/'
s025 = 'd://z/master/comvis/color_indices/outputs/combined_data_025.csv'
s050 = 'd://z/master/comvis/color_indices/outputs/combined_data_050.csv'
s075 = 'd://z/master/comvis/color_indices/outputs/combined_data_075.csv'
s100 = 'd://z/master/comvis/color_indices/outputs/combined_data_100.csv'

"""LOAD DATA"""
def load_data(data):
    data = pd.read_csv(data)
    return data
dt_s25 = load_data(s025)
dt_s50 = load_data(s050)
dt_s75 = load_data(s075)
dt_s100 = load_data(s100)

"""COMBINE DATA"""
def combine_data(*dataframes):
    combined = pd.concat(dataframes, ignore_index=True)
    return combined
combined_data = combine_data(dt_s25, dt_s50, dt_s75, dt_s100)

"""LABEL NAME"""
label_name = ['25%', '50%', '75%', '100%']

"""EXTRACT VARIABLE DATA"""
def variable_data(data):
    # --- Label
    label = data.values[:,0].astype('uint8')
    # --- Data
    spectra = data.iloc[:,1:]#.astype('float')
    # --- Feature
    cols = list(data.columns[1:])
    return label, spectra, cols
label, data, feature = variable_data(combined_data)

print(f'Label: {label.shape}, Data: {data.shape}, Feature: {len(feature)}')
combine_data = pd.DataFrame(combined_data, columns=feature)
feature_name = combine_data.columns.tolist()
print(combined_data.info())

dt_s25.describe().round(4).to_csv('d://z/master/comvis/color_indices/outputs/data_desc_s025.csv')
dt_s50.describe().round(4).to_csv('d://z/master/comvis/color_indices/outputs/data_desc_s050.csv')
dt_s75.describe().round(4).to_csv('d://z/master/comvis/color_indices/outputs/data_desc_s075.csv')
dt_s100.describe().round(4).to_csv('d://z/master/comvis/color_indices/outputs/data_desc_s100.csv')

"""CORRELATION MATRIX"""
corr = combined_data.iloc[:, 1:].corr(numeric_only=True)
plt.figure(figsize=(25, 24))
ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5,
            annot_kws={"size": 10}, cbar_kws={"shrink": .5})
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xlabel('Features', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
#plt.title('Correlation Matrix')
plt.savefig(outputPath + 'correlation_matrix.png')

"""Scaled combined_data"""
# Standardize features, except first column (label)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(combined_data.iloc[:, 1:])
df_scaled = pd.DataFrame(df_scaled ,columns=combined_data.columns[1:])
# save scaled dataframe with label inserted
df_scaled.insert(0, 'label', combined_data['label'])
df_scaled.to_csv(outputPath + 'combined_data_scaled.csv', index=False)


"""PRINCIPAL COMPONENT ANALYSIS FOR DIMENSIONALITY REDUCTION"""
"""Step 1 - Data scaling"""
x_scaled = StandardScaler().fit_transform(data)

"""Step 2 - Find the covariance and correlation matrices"""
mean_vector = np.mean(x_scaled, axis=0)
covmat = (x_scaled - mean_vector).T.dot((x_scaled - mean_vector)) / (x_scaled.shape[0] - 1)
# another way
covmat_ = np.cov(x_scaled, rowvar=False)

"""Step 3 - Eigen decomposition"""
# Eigen from covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covmat)
# Eigen from correlation matrix
cormat = np.corrcoef(x_scaled.T)
eigenvalues_, eigenvectors_ = np.linalg.eig(cormat)

"""Step 4 - Sort the eigenvalues and eigenvectors"""
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x_scaled: x_scaled[0], reverse=True)

"""Step 5 - Step 5 - Choose PCs by selecting top k eigvecs"""
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for a in eigen_pairs:
    if(a[0]>0.1):
        print('Sorted Eigenvalues: {}'.format(a[0]))
# For further usage
eig_vals_sorted = np.array([x_scaled[0] for x_scaled in eigen_pairs])
eig_vecs_sorted = np.array([x_scaled[1] for x_scaled in eigen_pairs])

eigval_total = sum(eigenvalues)
explained_variance = [(i/eigval_total) for i in sorted(eigenvalues, reverse=True)]
explained_variance = np.round(explained_variance, 3).real

for b in explained_variance:
    if (b>(0.1/100)):
         print('Explained Variance: {}%'.format(round(b*100, 3)))

cum_explained_variance = np.cumsum(explained_variance)
cum_explained_variance = np.round(cum_explained_variance, 3)

for c in cum_explained_variance:
    if (c<0.99):
        print('Cumulative Explained Variance: {}%'.format(round(c*100, 3)))

"""Step 6 - Project the data onto the new feature space"""
k = 3  # Number of principal components
W = eig_vecs_sorted[:k, :]
W = np.array(W, dtype=float)

X_proj = x_scaled.dot(W.T)

pc_list = ['PC'+ str(i) for i in list(range(1, k+1))]

"""PCA Loading or correlation coefficients"""
loadings = pd.DataFrame.from_dict(dict(zip(pc_list, W)))
#print(loadings.head())
loading_csv_path = outputPath + 'pca_loadings.csv'
loadings.to_csv(loading_csv_path, index=False)

"""PCA Scores"""
score = pd.DataFrame.from_dict(dict(zip(pc_list, X_proj)))
df_score = pd.DataFrame(X_proj, columns= pc_list, index=label) # change label
df_score = df_score[df_score.columns[0:k]]
df_score.to_csv(outputPath + 'pca_scores.csv', index=False)
#print(df_score.head(5))

"""Scree Plot"""
plt.figure(figsize=(7, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
#plt.plot(range(1, len(cum_explained_variance) + 1), cum_explained_variance, marker='o', linestyle='--', color='orange')
#plt.title('Scree Plot')
plt.xlabel('Number of Principal Components (PC)')
plt.ylabel('Variance Explained')
plt.grid()
#plt.xticks(range(1, len(explained_variance) + 1))
plt.ylim(-0.05, 0.85)
#plt.legend(['Individual Explained Variance', 'Cumulative Explained Variance'], loc='center right')
plt.savefig(outputPath + 'pca_scree_plot.png')

"""PCA score plot"""
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_score, x='PC2', y='PC3', hue=label, palette='viridis', s=100, edgecolor='k', alpha=0.7)
#plt.title('PCA Score Plot')
plt.xlabel('PC2 ({})%'.format(round(explained_variance[1]*100, 2)), size=12).set_color('black')
plt.ylabel('PC3 ({})%'.format(round(explained_variance[2]*100, 2)), size=12).set_color('black')
plt.axhline(y=0.0, color='black', linestyle='dashed', alpha = 1.0)
plt.axvline(x=0.0, color='black', linestyle='dashed', alpha = 1.0)
plt.grid()
#plt.legend(title='irrigation levels', loc='upper right', fontsize=10, title_fontsize=10)
#plt.tight_layout()
plt.savefig(outputPath + 'pca_score_plot.png')

"""PCA biplot"""
fig, ax = plt.subplots(figsize=(6,5))
for i, feature in enumerate(feature):
    ax.arrow(0, 0, W[0,i], W[1,i], head_width = 0.02, head_length = 0.02, color = 'black')
    ax.text(W[0,i]*1.15, W[1,i]*1.15, feature, color = 'black', fontsize=10).set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='gray'))
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title('2D PCA Biplot')
ax.axhline(y=0.0, color='black', linestyle='dashed', alpha = 1.0) # label = 'Horizontal Line ')
ax.axvline(x=0.0, color='black', linestyle='dashed', alpha = 1.0) # label = 'Vertical Line ')
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')
yabs_max = abs(max(ax.get_ylim(), key=abs))
ax.set_ylim(ymin=-(yabs_max+(yabs_max/3)), ymax=(yabs_max+(yabs_max/3)))
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-(xabs_max+(xabs_max/3)), xmax=(xabs_max+(xabs_max/3)))
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.set_facecolor("white")
plt.savefig(outputPath + 'pca_biplot.png')

"""PCA Loadings Plot"""
plt.figure(figsize=(8, 6))
plt.matshow(W[0:3,:], cmap='viridis')
plt.yticks([0,1,2], ["PC1 ({})%".format(round(explained_variance[0]*100, 2)), 
                     "PC2 ({})%".format(round(explained_variance[1]*100, 2)), 
                     "PC3 ({})%".format(round(explained_variance[2]*100, 2))], size=12)
plt.colorbar()
plt.xticks(range(len(feature_name)),
           feature_name, rotation=90, ha='left', size=12)
plt.xlabel("Features", size=12)
plt.ylabel("Loadings", size=12, rotation=90)
#plt.tight_layout()
plt.savefig(outputPath + 'pca_loadings.png')

"""3D PCA score plot"""
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
scatter = ax.scatter(df_score['PC1'], df_score['PC2'], df_score['PC3'], c=label, edgecolors='k', cmap='viridis', s=50)
ax.set(
    title='3D PCA Score Plot',
    xlabel='PC1 ({}%)'.format(round(explained_variance[0]*100, 2)),
    ylabel='PC2 ({}%)'.format(round(explained_variance[1]*100, 2)),
    zlabel='PC3 ({}%)'.format(round(explained_variance[2]*100, 2))
)
import matplotlib.patches as mpatches
unique_labels = np.unique(label)
legend_handles = [mpatches.Patch(color=plt.cm.viridis(i / max(unique_labels)),
                                 label=label_name[i]) for i in unique_labels]
ax.legend(handles=legend_handles, title="irrigation treatments", loc='upper right', fontsize=10)
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.zaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.tick_params(axis='z', colors='black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.set_facecolor("white")
plt.savefig(outputPath + 'pca_3d_score_plot.png')

"""STATISTICAL ANALYSIS"""
# One-way ANOVA test
from scipy.stats import f_oneway
col = 32  # Change this to the index of the column you want to test
group1 = dt_s25.iloc[:10, col].values.flatten()
group2 = dt_s50.iloc[:10, col].values.flatten()
group3 = dt_s75.iloc[:10, col].values.flatten()
group4 = dt_s100.iloc[:10, col].values.flatten()
f_stat, p_value = f_oneway(group1, group2, group3, group4)
print(f"F-statistic: {f_stat}, P-value: {p_value}")
# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There are significant differences between groups.")
else:
    print("Fail to reject the null hypothesis: No significant differences found.")

# Perform Tukey's HSD test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Combine all values into one array
data = list(group1) + list(group2) + list(group3) + list(group4)
# Create group labels
labels = (['S25'] * len(group1) +
          ['S50'] * len(group2) +
          ['S75'] * len(group3) +
          ['S100'] * len(group4))
tukey_result = pairwise_tukeyhsd(endog=data, groups=labels, alpha=0.05)
print(tukey_result)
tukey_result.plot_simultaneous()
plt.title('Tukey HSD Test Results')
# Combine data
import scikit_posthocs as sp
data = np.concatenate([group1, group2, group3, group4])
groups = (['S25'] * len(group1) +
          ['S50'] * len(group2) +
          ['S75'] * len(group3) +
          ['S100'] * len(group4))
# Create DataFrame
df = pd.DataFrame({'score': data, 'group': groups})
dunn_result = sp.posthoc_dunn(df, val_col='score', group_col='group', p_adjust='bonferroni')
dunn_result.to_csv(outputPath + 'dunn_posthoc_results.csv', index=True)


"""HIERARCHICAL CLUSTER ANALYSIS FOR ITEMS"""
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import pdist

# distance between items (rows)
item_distances = pdist(df_scaled.iloc[:, 1:], metric='euclidean') # 'euclidean', 'cityblock', 'cosine', 'correlation'
# save distance matrix
dist_matrix = pd.DataFrame(scipy.spatial.distance.squareform(item_distances), columns=combined_data.index, index=combined_data.index)
dist_matrix.to_csv(outputPath + 'hierarchical_item_distance_matrix.csv', index=True)
# Visualize the distance matrix
plt.figure(figsize=(10, 8))
sns.heatmap(dist_matrix, annot=False, fmt=".2f", cmap='viridis', linewidths=0.5,
            annot_kws={"size": 10}, cbar_kws={"shrink": .5})
plt.title('Item Distance Matrix')
plt.xlabel('Items', fontsize=16)
plt.ylabel('Items', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig(outputPath + 'hierarchical_item_distance_matrix.png')

# visualize dendrogram for items
plt.figure(figsize=(15, 8))
Z_items = linkage(item_distances, method='ward') # 'ward', 'single',
# Get variance or error sum of squares, SSE, of ward method
SSE = sum((Z_items[:, 2] ** 2) * Z_items[:, 3])
print(f'Sum of squared distances (SSE) for items: {SSE}')
# create a DataFrame from the linked matrix.
linkage_df_items = pd.DataFrame(Z_items, columns=['Cluster1', 'Cluster2', 'Distance', 'SampleCount'])
linkage_df_items.to_csv(outputPath + 'hierarchical_item_clustering_linkage.csv', index=False)
print(linkage_df_items.head(10))
# Find center of clusters
# https://stackoverflow.com/questions/66570385/finding-the-centroid-of-each-cluster-from-hierarchical-clustering-in-python
# Dendrogram
plt.figure(figsize=(20, 8))
dendrogram(Z_items, 
           orientation='top',
           truncate_mode='lastp',  # 'lastp' or 'level'
           p=90,  # Show only the last p merged clusters
           leaf_font_size=12,
           color_threshold=0.5,
           labels = combined_data.index.tolist(),
           distance_sort='descending',
           show_leaf_counts=True)
plt.xlabel('Items')
plt.ylabel('SSE')
plt.savefig(outputPath + 'hierarchical_item_dendrogram.png')
# Form item clustering (e.g., target 4 clusters)
num_clusters = 4    # Adjust the number of clusters as needed
clusters_items = fcluster(Z_items, num_clusters, criterion='maxclust')
item_clusters = pd.DataFrame({'Item': combined_data.index, 'Cluster': clusters_items})
print("\nItem Clusters:")
print(item_clusters)
# Find the number of items in each cluster
cluster_counts = item_clusters['Cluster'].value_counts().sort_index()
print("\nNumber of items in each cluster:")
print(cluster_counts)
# Visualize item clusters
plt.figure(figsize=(10, 6))
for cluster in item_clusters['Cluster'].unique():
    plt.scatter(item_clusters[item_clusters['Cluster'] == cluster]['Item'],
                [cluster] * item_clusters[item_clusters['Cluster'] == cluster].shape[0],
                label=f'Cluster {cluster}')
plt.title('Item Clusters')
plt.xlabel('Items')
plt.ylabel('Cluster')
plt.xticks(rotation=90, fontsize=10)
#plt.legend(loc='upper left')
plt.savefig(outputPath + 'hierarchical_item_clusters.png')

"""HIERARCHICAL CLUSTER ANALYSIS FOR VARIABLES"""
# Calculate pairwise distances between variables or features (columns)
# Using correlation as a similarity measure, and then converting to distance
# distance metric = 1 - Pearson correlation coefficient
# Transpose for feature-wise clustering
corr_data = df_scaled.iloc[:, 1:].corr(method='pearson').abs() # # Using absolute correlation for distance
# Save correlation matrix
corr_data.to_csv(outputPath + 'hierarchical_correlation_matrix.csv', index=True)
# Visualize the correlation matrix
plt.figure(figsize=(25, 24))
sns.heatmap(corr_data, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5,
            annot_kws={"size": 10}, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix for Clustering')
plt.xlabel('Features', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig(outputPath + 'hierarchical_correlation_matrix.png')

# Compute the distance matrix
dist_matrix = (1 - corr_data)
variable_distances = pdist(dist_matrix.T, metric='euclidean')
# Distance matrix as DataFrame
dist_df = pd.DataFrame(scipy.spatial.distance.squareform(variable_distances), columns=corr_data.columns, index=corr_data.columns)
dist_df.to_csv(outputPath + 'hierarchical_euclidean_distance_matrix.csv', index=True)
# Visualize the distance matrix
plt.figure(figsize=(25, 24))
sns.heatmap(dist_df, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5,
            annot_kws={"size": 10}, cbar_kws={"shrink": .5})
plt.title('Euclidean Distance Matrix')
plt.xlabel('Features', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig(outputPath + 'hierarchical_euclidean_distance_matrix.png')

# Similarity matrix as DataFrame
sim_df = pd.DataFrame(1 - scipy.spatial.distance.squareform(variable_distances), columns=corr_data.columns, index=corr_data.columns)
sim_df.to_csv(outputPath + 'hierarchical_similarity_matrix.csv', index=True)
print(sim_df.head())
# Visualize the similarity matrix
plt.figure(figsize=(25, 24))
sns.heatmap(sim_df, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5,
            annot_kws={"size": 10}, cbar_kws={"shrink": .5})
plt.title('Similarity Matrix')
plt.xlabel('Features', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig(outputPath + 'hierarchical_similarity_matrix.png')

"""Hierarchical Clustering"""
# Perform hierarchical clustering
Z = linkage(variable_distances, method='ward') # 'ward', 'single', 'complete', 'average', # 'ward' minimizes variance within clusters
# create a DataFrame from the linked matrix.
linkage_df = pd.DataFrame(Z, columns=['Cluster1', 'Cluster2', 'Distance', 'SampleCount'])
linkage_df.to_csv(outputPath + 'hierarchical_clustering_linkage.csv', index=False)
print(linkage_df.head(10))

# Visualize the Dendrogram
plt.figure(figsize=(15, 8))
dendrogram(Z, 
           orientation='top',
           truncate_mode='lastp',  # 'lastp' or 'level'
           p=90,  # Show only the last p merged clusters
           leaf_font_size=12,
           color_threshold=0.5,
           labels = corr_data.columns.tolist(),
           distance_sort='descending',
           show_leaf_counts=True)
#plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.savefig(outputPath + 'hierarchical_clustering_dendrogram.png')

# Form feature clustering (e.g., target 4 clusters)
num_clusters = 4    # Adjust the number of clusters as needed
clusters = fcluster(Z, num_clusters, criterion='maxclust')
feature_clusters = pd.DataFrame({'Feature': corr_data.columns, 'Cluster': clusters})
print("\nFeature Clusters:")
print(feature_clusters)

# Visualize feature clusters
plt.figure(figsize=(10, 6))
for cluster in feature_clusters['Cluster'].unique():
    plt.scatter(feature_clusters[feature_clusters['Cluster'] == cluster]['Feature'],
                [cluster] * feature_clusters[feature_clusters['Cluster'] == cluster].shape[0],
                label=f'Cluster {cluster}')
plt.title('Feature Clusters')
plt.xlabel('Features')
plt.ylabel('Cluster')
plt.xticks(rotation=90, fontsize=10)
#plt.legend(loc='upper left')
plt.savefig(outputPath + 'hierarchical_feature_clusters.png')

# Select Representative Features (example: pick first feature from each cluster)
selected_features = []
for cluster_id in sorted(feature_clusters['Cluster'].unique()):
    features_in_cluster = feature_clusters[feature_clusters['Cluster'] == cluster_id]['Feature'].tolist()
    # Example: Select the first feature in each cluster
    selected_features.append(features_in_cluster[0])

print(f"\nSelected Features: {selected_features}")
# https://readmedium.com/feature-selection-with-hierarchical-clustering-for-interpretable-models-a091802f24e0

from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import silhouette_score
# Choosing the Optimal Number of Clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    preds = clusterer.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, preds)
    silhouette_scores.append(score)
# Plot silhouette scores
plt.figure(figsize=(6, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.savefig(outputPath + 'hierarchical_silhouette_scores.png')

# Apply Agglomerative Clustering based on the dendrogram
agglo_clusters = AgglomerativeClustering(n_clusters=4).fit_predict(x_scaled)
# Plot the clustered data
plt.figure(figsize=(8, 5))
scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=agglo_clusters, cmap='viridis', s=30)
plt.title('Hierarchical Clustering on RGB color indices (PCA reduced)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(scatter, ticks=[0, 1, 2])
plt.savefig(outputPath + 'hierarchical_clustering_pca.png') 
#https://hex.tech/blog/comparing-density-based-methods/#comparing-the-methods

# Plot 3D clustered data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], c=agglo_clusters, edgecolors='k', cmap='viridis', s=50)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('3D Hierarchical Clustering')
plt.savefig(outputPath + 'hierarchical_clustering_3d_pca.png')

"""K-MEANS CLUSTERING"""
# https://realpython.com/k-means-clustering-python/
from sklearn.cluster import KMeans
true_labels = label
# K-Means clustering
kmeans = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(X_proj)

# The lowest SSE value
sse = kmeans.inertia_
print(f'Sum of squared distances (Inertia): {sse}')
# Final locations of the centroid
centers = kmeans.cluster_centers_
print(f'Centroids: {centers}')
# Get the cluster labels
predicted_labels = kmeans.labels_
print(f'Predicted labels: {predicted_labels}')
# The number of iterations required to converge
n_iter = kmeans.n_iter_
print(f'Number of iterations: {n_iter}')

# Plot true labels with predicted labels
fig = plt.figure(figsize=(6, 5))
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=predicted_labels, cmap='viridis', s=30)
#plt.scatter(X_proj[:, 0], X_proj[:, 1], c=true_labels, cmap='viridis', s=30)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering on RGB color indices (PCA reduced)")
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(outputPath + 'kmeans_clustering.png')
#plt.show()

# Plot 3D clustered data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], c=predicted_labels, edgecolors='k', cmap='viridis', s=50)
#scatter = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='X', s=200, label='Centroids')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('3D K-Means Clustering')
plt.savefig(outputPath + 'kmeans_clustering_3d.png')

# Elbow method to find the optimal number of clusters
inertia = []
for n_clusters in range(1, 47):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_proj)
    inertia.append(kmeans.inertia_)
# Plot after collecting all inertia values
plt.figure(figsize=(6, 5))
plt.plot(range(1, 47), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.grid()
plt.savefig(outputPath + 'kmeans_elbow_method.png')
#plt.show()

# Silhouette analysis
from sklearn.metrics import silhouette_score
# A list holds the silhouette coefficients for each k
silhouette_coefficients = []
# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 47):
    kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X_proj)
    score = silhouette_score(X_proj, kmeans.labels_)
    silhouette_coefficients.append(score)

# Plot silhouette scores
plt.figure(figsize=(6, 5))
plt.plot(range(2, 47), silhouette_coefficients, marker='o')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.savefig(outputPath + 'kmeans_silhouette_scores.png')

# Find the number of items in each cluster
cluster_counts = pd.Series(predicted_labels).value_counts().sort_index()
print("\nNumber of items in each K-Means cluster:")
print(cluster_counts)

"""Density-based spatial clustering application with noise (DBSCAN)"""


"""Spectral Clustering"""
# https://www.geeksforgeeks.org/machine-learning/ml-spectral-clustering/