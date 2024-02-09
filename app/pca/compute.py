from sklearn.decomposition import PCA


class custom_pca:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, data):
        self.pca.fit(data)

    def transform(self, data):
        return self.pca.transform(data)
