from sklearn.decomposition import PCA


def compute_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    ret = pca.transform(data)
    del pca
    return ret
