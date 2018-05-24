from techniques.pca import PCA
def compute_mapping(A, type, n_components):
    if(type == "PCA"):
        pca_model = PCA(n_components = n_components)
        pca_model.fit(A)
        mapping = pca_model.components_
        mappedA = pca_model.transform(A)
    else:
        print("Unknown dimensionality reduction technique.")

    return mappedA, mapping