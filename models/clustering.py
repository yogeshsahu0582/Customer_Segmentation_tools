import pandas as pd
from sklearn.cluster import KMeans

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):

    return data

def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(data)
    return data, kmeans
