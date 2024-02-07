from sklearn.decomposition import PCA


def compute_pca(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)
