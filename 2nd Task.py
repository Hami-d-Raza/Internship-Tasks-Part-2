# ====================================================
# Task 2: Customer Segmentation Using Unsupervised Learning
# ====================================================
# Objective:
# Cluster customers based on spending habits and propose marketing strategies tailored to each segment.
#
# Dataset: Mall Customers Dataset
# Instructions:
# ● Conduct Exploratory Data Analysis (EDA)
# ● Apply K-Means Clustering to segment customers
# ● Use PCA or t-SNE to visualize the clusters
# ● Suggest relevant marketing strategies for each identified segment
# ====================================================

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------
# Step 2: Load Dataset
# -------------------------
# Use raw string or forward slashes to avoid escape sequence errors
df = pd.read_csv(r"DS_Tasks 2\Mall_Customers.csv")

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------
# Step 3: Exploratory Data Analysis (EDA)
# -------------------------
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Gender Distribution (column name is "Genre")
sns.countplot(x="Genre", data=df)
plt.title("Gender Distribution")
plt.show()

# Age vs Spending Score
sns.scatterplot(x="Age", y="Spending Score (1-100)", data=df, hue="Genre")
plt.title("Age vs Spending Score")
plt.show()

# Income vs Spending Score
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df, hue="Genre")
plt.title("Annual Income vs Spending Score")
plt.show()

# -------------------------
# Step 4: Feature Selection & Scaling
# -------------------------
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Step 5: Find Optimal Clusters (Elbow Method)
# -------------------------
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.show()

# -------------------------
# Step 6: Apply K-Means Clustering
# -------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# Cluster Visualization (2D - Income vs Spending Score)
plt.figure(figsize=(8,6))
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)",
                hue="Cluster", data=df, palette="Set1")
plt.title("Customer Segments (K-Means)")
plt.show()

# -------------------------
# Step 7: Dimensionality Reduction (PCA & t-SNE)
# -------------------------
# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = pca_result[:,0], pca_result[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="Set2")
plt.title("PCA Visualization of Clusters")
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(X_scaled)
df["tSNE1"], df["tSNE2"] = tsne_result[:,0], tsne_result[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x="tSNE1", y="tSNE2", hue="Cluster", data=df, palette="Set3")
plt.title("t-SNE Visualization of Clusters")
plt.show()

# -------------------------
# Step 8: Marketing Strategies for Each Cluster
# -------------------------
print("\nSuggested Marketing Strategies:")
strategies = {
    0: "Cluster 0: Low Income, Low Spending → Cost-sensitive customers. Offer discounts and budget-friendly deals.",
    1: "Cluster 1: High Income, Low Spending → Premium customers. Use loyalty programs, exclusive offers to increase engagement.",
    2: "Cluster 2: Moderate Income, High Spending → Trendsetters. Promote new arrivals, personalized marketing campaigns.",
    3: "Cluster 3: High Income, High Spending → Luxury seekers. Target with premium products, VIP services, luxury experiences.",
    4: "Cluster 4: Young customers with varied spending → Attract with social media campaigns, gamified rewards, influencer marketing."
}

for cluster, strategy in strategies.items():
    print(strategy)
