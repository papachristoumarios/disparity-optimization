import pandas as pd
import json
import numpy as np
import os
from sklearn.decomposition import PCA

datasets = ['DE', 'ES', 'FR', 'PTBR', 'RU']

for dataset in datasets:
    print(f'preprocessing {dataset}')
    musae_edges = pd.read_csv(f'{dataset}/musae_{dataset}_edges.csv')
    musae_labels = pd.read_csv(f'{dataset}/musae_{dataset}_target.csv')
    musae_features = json.load(open(f'{dataset}/musae_{dataset}_features.json'))

    musae_labels = musae_labels[['new_id', 'mature']].sort_values(by='new_id')
    musae_labels['mature'] = musae_labels['mature'].astype(int)
    musae_labels['mature'] = 2 * musae_labels['mature'] - 1

    os.makedirs(f'../twitch-{dataset}', exist_ok=True)

    musae_labels_output = f'../twitch-{dataset}/opinions.txt'
    musae_labels.to_csv(musae_labels_output, sep='\t', header=False, index=False)

    musae_edges_output = f'../twitch-{dataset}/edges.txt'
    musae_edges.to_csv(musae_edges_output, sep='\t', header=False, index=False)

    max_features = 0

    for node_id, features in musae_features.items():
        temp = max(features)
        max_features = max(max_features, temp)

    features_matrix = np.zeros((len(musae_features), max_features + 1))
    for node_id, features in musae_features.items():
        for feature_id in features:
            features_matrix[int(node_id), int(feature_id)] = 1.0

    np.save(f'../twitch-{dataset}/features.npy', features_matrix)

    # pca the features matrix
    pca = PCA(n_components=32, random_state=0)
    features_matrix = pca.fit_transform(features_matrix)
    np.save(f'../twitch-{dataset}/embeddings.npy', features_matrix)

    

