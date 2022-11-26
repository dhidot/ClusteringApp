import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle

df = pd.read_csv('transfusion.data')

# print(df.head(10))
# print(df.shape)

# # Mengubah tipe data atribut class menjadi string
# df['class'] = df['class'].apply(str)
#
# # Ubah value dari class menjadi yes dan no
# df.loc[df["class"] == "1", "class"] = 'yes'
# df.loc[df["class"] == "0", "class"] = 'no'

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
# print(kfit.cluster_centers_)
# print(kfit.labels_)
# print(kfit.inertia_)
# print(kfit.n_iter_)
# print(kfit.n_clusters)
# print(kfit.n_features_in_)
# print(kfit.feature_names_in_)

centroids = kmeans.cluster_centers_
print(centroids)
print(centroids.shape)
#
# # Menyimpan model ke dalam file pickle secara local
# with open('centroids.pkl', 'wb') as f:
#     pickle.dump(centroids, f)
#     print('Centroids tersimpan ke dalam file pickle')
