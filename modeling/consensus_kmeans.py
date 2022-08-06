# packages
import biosppy.clustering as bioc
import biosppy.plotting as biop
import biosppy.metrics as biom
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster import hierarchy
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt
# local
from kmeans import KmeansClustering
from gaussian_mixture import GaussianMixtureClustering


class ConsensusKmeans:

    def __init__(self, kmin=None, kmax=None, n_ensemble=100):
        self.kmin = kmin
        self.kmax = kmax
        self.n_ensemble = n_ensemble

    def ensemble(self, data):

        N = len(data)

        if self.kmin is None:
            self.kmin = int(round(np.sqrt(N) / 2.))

        if self.kmax is None:
            self.kmax = int(round(np.sqrt(N)))

        grid = {
            'n_clusters': np.random.randint(low=self.kmin, high=self.kmax, size=self.n_ensemble)
        }

        # ensemble, = bioc.create_ensemble(data.to_numpy(), fcn=bioc.kmeans, grid=grid)
        # return ensemble

        metrics = ['EUCLIDEAN', 'EUCLIDEAN_SQUARE', 'MANHATTAN', 'CHEBYSHEV', 'CANBERRA', 'CHI_SQUARE']
        grid = ParameterGrid(grid)

        # run kmeans for each ensemble
        ensemble = []
        
        for params in grid:
            # algs = ['kmeans', 'gaussian']
            # metric = np.random.choice(metrics, 1, p=[0.4, 0.5, 0.025, 0.025, 0.025, 0.025])
            # alg = np.random.choice(algs, 1, p=[0.7, 0.3])
            # if alg == 'kmeans':
            #    k_means = KmeansClustering(data=data, init='random', metric='EUCLIDEAN_SQUARE', **params)
            # else:
            #     k_means = GaussianMixtureClustering(covariance_type='spherical', **params)
            # print(k_means)
            k_means = KmeansClustering(data=data, init='random', metric='EUCLIDEAN', **params)
            y_pred = k_means.fit_predict(data)
            ensemble.append(bioc._extract_clusters(y_pred))
        return ensemble

    def coassoc_matrix(self, ensemble, data_size, path=None, show=False):
        coassoc, = bioc.create_coassoc(ensemble, data_size)
        plt.imshow(coassoc, interpolation='nearest')
        if path is not None:
            plt.savefig(path + '.png')
        if show:
            plt.show()
        return coassoc

    def coassoc_partition(self, coassoc, k, linkage):
        clusters, = bioc.coassoc_partition(coassoc, k, linkage)

        # # convert coassoc to condensed format, dissimilarity
        # mx = np.max(coassoc)
        # D = biom.squareform(mx - coassoc)

        # # build linkage
        # Z = hierarchy.linkage(D, method=linkage)
        # plt.figure()
        # dn = hierarchy.dendrogram(Z)
        # plt.show()

        # labels = hierarchy.fcluster(Z, k, 'maxclust')

        # # get cluster indices
        # clusters = bioc._extract_clusters(labels)

        return clusters

    def visualize_clusters(self, data, clusters, path=None, show=False):
        biop.plot_clustering(data.to_numpy(), clusters, path, show)

        # determine number of clusters
        keys = list(clusters)
        n_rows = len(data)
        y_pred = np.ones((n_rows,), dtype=int)

        for k in keys:
            y_pred[clusters[k]] = k
            # if i == 0:
            #     axis_x = data.iloc[clusters[k], :]

        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot(111, projection='3d')
        sc = axis.scatter(
            data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=y_pred
        )
        axis.set_xlabel(data.columns[0], fontsize=10)
        axis.set_ylabel(data.columns[1], fontsize=10)
        axis.set_zlabel(data.columns[2], fontsize=10)
        plt.legend(*sc.legend_elements(), loc=1, title='Clusters')
        if path is not None:
            plt.savefig(path + '.png')
        if show:
            plt.show()
        return y_pred

    def evaluate_clusters(self, data, y_pred, path=None, show=False):
        clusters = np.unique(np.array(y_pred)) 

        if len(clusters) == 1:
            c_h_score = 'Only one cluster found'
            d_b_score = 'Only one cluster found'
            s_score = 'Only one cluster found'
        else: 
            c_h_score = calinski_harabasz_score(data, y_pred)
            d_b_score = davies_bouldin_score(data, y_pred)
            s_score = silhouette_score(data, y_pred)
        
        if path:
            with open(path + 'evaluation.txt', 'a+') as f:
                for _, v in enumerate(clusters):
                    n = len(y_pred[y_pred == v])
                    f.write('N instances belonging to cluster {}: {} \n'.format(v, n)) 
                f.write('Calinski score: {} \n'.format(c_h_score))
                f.write('Davies-Bouldin score: {} \n'.format(d_b_score))
                f.write('Silhouette score: {} \n \n'.format(s_score))
        if show:
            for _, v in enumerate(clusters):
                n = len(y_pred[y_pred == v])
                print('N instances belonging to cluster {}:'.format(v), n) 
            print('Calinski score:', c_h_score)
            print('Davies-Bouldin score:', d_b_score)
            print('Silhouette score:', s_score, '\n')
