# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages
2.Insert the dataset to perform the k - means clustering
3.perform k - means clustering on the dataset
4.Then print the centroids and labels
5.Plot graph and display the clusters
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SRISHANTH J
RegisterNumber:  212223240160
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Mall_Customers_EX8.csv")
data
X=data[["Annual Income (k$)","Spending Score (1-100)"]]
X
plt.figure(figsize=(4,4))
plt.scatter(data["Annual Income (k$)"],data["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
k =5
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b','c','m']
for i in range(k):
  cluster_points = X[labels==i]
  plt.scatter(cluster_points["Annual Income (k$)"],cluster_points["Spending Score (1-100)"],
              color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:,0],centroids[:,1],marker="*",s=200,color='k',label='Centroids')
plt.title("K- means Clustering")
plt.xlabel("Annual Incme (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
```

## Output:
- DATASET:
![Screenshot 2024-04-23 084449](https://github.com/srishanth2006/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150319470/c48e0966-9471-4497-a7c9-a17809588a77)
- CENTROID AND LABEL VALUES:
![Screenshot 2024-04-23 084511](https://github.com/srishanth2006/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150319470/ef769f22-0346-4c85-911a-a11b88719299)
- CLUSTERING:
- ![Screenshot 2024-04-23 084527](https://github.com/srishanth2006/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150319470/80e97e88-c784-4158-ba7f-20ae8ed85141)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
