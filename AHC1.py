from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


def hierarchical_clustering(data, number_cluster, connect):
    # computing dendrogram structure and show in a plot
    # plt.figure(figsize=(10, 7))
    # plt.title("Customer Dendograms")
    # dend = sch.dendrogram(sch.linkage(data, method='ward'))

    # main computing
    cluster = AgglomerativeClustering(n_clusters=number_cluster, connectivity=connect,
                                      linkage='ward').fit(data)
    # cluster.fit_predict(data)

    # cluster labels
    final_labels = cluster.labels_

    # compute Calinski-Harabasz Index - fast computation
    eval_metric = metrics.calinski_harabasz_score(data, final_labels)
    print('Evaluation Clustering By Calinski-Harabasz Index for this picture is : ', eval_metric)

    # compute Silhouette Score - ignore it for long time computation
    # silhouette_avg = silhouette_score(data, final_labels)
    # print("For n_clusters =", number_cluster,
    #       "The average silhouette_score is :", silhouette_avg)

    return final_labels, eval_metric

