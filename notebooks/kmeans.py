import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
mpl.style.use("seaborn")
mpl.rcParams['legend.frameon'] = 'True'


def kmeans_topic_clustering(model,
                            vectors,
                            n_clusters=5,
                            cluster_size=12000,
                            fontsize=15):
  """
  input : model, vector repr. of the texts
  output: a segregation of topic clusters
  """

  model_output = model.fit_transform(vectors)
  tsne_model = TSNE(n_components=2, random_state=42)  # 2 components
  model_output_tsne = tsne_model.fit_transform(model_output)

  KM = KMeans(n_clusters=n_clusters, random_state=42)
  clusters = KM.fit_predict(model_output_tsne)
  centroids = KM.cluster_centers_

  x = model_output_tsne[:, 0]
  y = model_output_tsne[:, 1]

  # Plot
  colors = [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ]

  plt.figure(figsize=(8, 7))

  plt.scatter(x, y, c=[colors[c] for c in clusters], alpha=0.3)
  plt.scatter(centroids[:, 0],
              centroids[:, 1],
              marker='*',
              color="black",
              s=150)
  plt.scatter(centroids[:, 0],
              centroids[:, 1],
              marker='o',
              color=[colors[i] for i in range(len(centroids))],
              s=cluster_size,
              alpha=0.3)
  plt.xlabel('Component 1', fontsize=fontsize)
  plt.ylabel('Component 2', fontsize=fontsize)
  plt.xticks(fontsize=fontsize - 1)
  plt.yticks(fontsize=fontsize - 1)
  plt.title(f"Segregation of topic clusters in the documents",
            fontsize=fontsize)
  plt.show()