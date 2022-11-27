import pandas as pd
from sklearn.cluster import KMeans
import pickle

df = pd.read_csv('transfusion.data')

x = df[['frequency', 'monetary']]

# KMeans alqorithm
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0,
)

kfit = kmeans.fit(x)

centroids = kmeans.cluster_centers_
print(centroids)
print(centroids.shape)

# Menyimpan model ke dalam file pickle
with open('centroids.pkl', 'wb') as f:
    pickle.dump(centroids, f)
    print('Centroids tersimpan ke dalam file pickle')
