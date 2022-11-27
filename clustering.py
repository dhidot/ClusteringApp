import pandas as pd
from sklearn.cluster import KMeans
import pickle

df = pd.read_csv('transfusion.data')

x = df[['recency', 'frequency', 'monetary', 'time']]
# print(x.head(10))

# KMeans alqorithm
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=90,
)

kfit = kmeans.fit(x)

centroids = kmeans.cluster_centers_
print(centroids)
print(centroids.shape)

# Menyimpan model ke dalam file pickle secara local
with open('centroids.pkl', 'wb') as f:
    pickle.dump(centroids, f)
    print('Centroids tersimpan ke dalam file pickle')
