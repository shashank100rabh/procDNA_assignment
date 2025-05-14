import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the data
df = pd.read_feather('Clustering_Data.ftr')
print("Original data : ")
print(df.head())
# Selecting the  relevant features, as features like Address, name etc. are not usefrul for clustering
features = [
    'Seasonality_Segment', 'EA_Segment', 'Revenue_Bucket', 'Profit_Bucket',
    'Market_Share_Segment', 'Casino_Size_Segment', 'Market_Potential_Segment',
    'Churn_Segment', 'Competitiveness_Flag', 'Volume_Segment', 'Density_Segment',
    'Propensity'
]

# Subset the DataFrame
data = df[features].copy()

# Replace 'None', '-' and missing values with a placeholder (or drop if needed)
data.replace(['None', '-', None], np.nan, inplace=True)
data.dropna(inplace=True)  # Or use imputation as an alternative
print("data after selecting a subset and removing NaN values :")
#Encoding categorical values using OrdinalEncoder
encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data)

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)




#Elbow Method and Silhouette Score
inertia = []
silhouette = []
K_range = range(2, 10)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    preds = model.fit_predict(data_scaled)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(data_scaled, preds))

# Plot the elbow method and silhouette score
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('k'); plt.ylabel('Inertia')

plt.subplot(1,2,2)
plt.plot(K_range, silhouette, marker='o', color='green')
plt.title('Silhouette Score')
plt.xlabel('k'); plt.ylabel('Score')
plt.tight_layout()
plt.show()



# After analyzing the elbow method and silhouette score, we choose k=5
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# View cluster assignments
print(data['Cluster'].value_counts())
print(data.head())




# REsucing dimensions with the PCA algorithm
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data_scaled)

# Add PCA results to the dataframe
data['PCA1'] = reduced_data[:, 0]
data['PCA2'] = reduced_data[:, 1]

# Visualizing the clusters using seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=60)
plt.title('Customer Clusters (via K-Means + PCA)', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()