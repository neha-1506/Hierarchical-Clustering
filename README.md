# Travel Review Segmentation using Hierarchical Clustering

# 1. Introduction

In today’s digital era, user reviews play a vital role in understanding customer preferences. Google travel reviews provide numerical ratings for various travel-related categories such as hotels, restaurants, parks, museums, and more.
Since these ratings reflect user interests, they can be used to segment users into meaningful groups.

This project applies Hierarchical Clustering, an unsupervised machine learning technique, to group users based on their travel review behavior.

---

# 2. Objective of the Project

* To analyze Google travel review ratings
* To preprocess the dataset for clustering
* To identify the optimal number of clusters using a dendrogram
* To segment users based on similar travel preferences

# 3. Importing Required Libraries

```python
# Numerical computation library
import numpy as np

# Data manipulation and analysis library
import pandas as pd

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Library to ignore warning messages
import warnings
warnings.filterwarnings('ignore')

# Display all columns without truncation
pd.set_option("display.max_columns", None)
```

### Explanation:

* `numpy` is used for numerical operations.
* `pandas` helps in reading, cleaning, and manipulating the dataset.
* `matplotlib` and `seaborn` are used for visual representation.
* `warnings` is used to suppress unnecessary messages for clean output.

---

# 4. Loading the Dataset

```python
# Reading the Google travel review dataset
df = pd.read_csv("google_review_ratings.csv", index_col=0)

# Displaying first five records
df.head()
```

### Explanation:

* The dataset is loaded using `read_csv()`.
* `index_col=0` sets the **User ID** as the index.
* Each row represents a unique user.
* Each column represents average ratings for a specific travel category.

---

# 5. Understanding Dataset Dimensions

```python
# Checking number of rows and columns
df.shape
```

### Explanation:

* The dataset contains 5456 rows (users) and 25 columns.
* This confirms sufficient data for clustering analysis.

---

# 6. Dataset Information

```python
# Checking column names, data types, and null values
df.info()
```

### Explanation:

* Most attributes are numerical (`float` type).
* One unnecessary column (`Unnamed: 25`) exists.
* Few columns contain missing values.
* Since clustering is distance-based, missing values must be treated.

---

# 7. Data Cleaning

## 7.1 Removing Irrelevant Column

```python
# Dropping column with mostly null values
df.drop("Unnamed: 25", axis=1, inplace=True)
```

### Explanation:

* This column does not contain meaningful information.
* Removing it improves data quality and accuracy.

---

## 7.2 Handling Missing Values

```python
# Checking missing values in each column
df.isnull().sum()

# Replacing missing values with column mean
df.fillna(df.mean(), inplace=True)
```

### Explanation:

* Hierarchical clustering cannot process missing values.
* Mean imputation maintains the overall rating distribution.
* Ensures dataset is complete for clustering.

---

# 8. Feature Scaling

```python
# Importing StandardScaler for normalization
from sklearn.preprocessing import StandardScaler

# Creating scaler object
scaler = StandardScaler()

# Scaling all numerical features
scaled_data = scaler.fit_transform(df)
```

### Explanation:

* Hierarchical clustering uses **Euclidean distance**.
* Features with higher values can dominate distance calculations.
* Scaling ensures all variables contribute equally.

---

# 9. Creating a Dendrogram

```python
# Importing hierarchical clustering library
import scipy.cluster.hierarchy as sch

# Plotting dendrogram
plt.figure(figsize=(15, 6))
sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.title("Dendrogram for Travel Review Segmentation")
plt.xlabel("Users")
plt.ylabel("Euclidean Distance")
plt.show()
```

### Explanation:

* A dendrogram visually represents cluster formation.
* `ward` linkage minimizes within-cluster variance.
* The height of vertical lines indicates distance between clusters.

---

# 10. Determining Optimal Number of Clusters

### Interpretation:

* Large vertical gaps in the dendrogram indicate cluster separation.
* Cutting the dendrogram at a suitable height gives **5 clusters**.
* Hence, the optimal number of clusters is chosen as **k = 5**.

---

# 11. Applying Hierarchical Clustering

```python
# Importing Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

# Creating clustering model
hc = AgglomerativeClustering(
    n_clusters=5,
    affinity='euclidean',
    linkage='ward'
)

# Fitting the model and predicting clusters
cluster_labels = hc.fit_predict(scaled_data)
```

### Explanation:

* Agglomerative clustering follows a **bottom-up approach**.
* Each data point starts as its own cluster.
* Clusters are merged based on minimum distance until k clusters remain.

---

# 12. Assigning Cluster Labels

```python
# Adding cluster labels to original dataset
df['Cluster'] = cluster_labels

# Displaying updated dataset
df.head()
```

### Explanation:

* Each user is assigned a cluster number.
* This enables easy identification of user groups.

---

# 13. Cluster Profiling and Interpretation

```python
# Calculating mean rating of each cluster
df.groupby('Cluster').mean()
```

### Explanation:

* Helps understand preferences of each cluster.
* Different clusters represent different travel interests.

### Example Interpretation:

* Cluster 0: Cultural and historical place lovers
* Cluster 1: Food and cafe enthusiasts
* Cluster 2: Entertainment and nightlife focused users
* Cluster 3: Nature and park lovers
* Cluster 4: Highly active travelers with high ratings

---

# 14. Final Conclusion

* Hierarchical clustering successfully segmented travel reviewers.
* Users were grouped based on similar rating behavior.
* Dendrogram helped identify optimal clusters.
* This segmentation helps in:

  * Personalized recommendations
  * Targeted marketing strategies
  * Travel trend analysis

---

# 15. Learning Outcome

* Understanding of hierarchical clustering
* Ability to interpret dendrograms
* Hands-on experience in customer segmentation
* Improved analytical and preprocessing skills

