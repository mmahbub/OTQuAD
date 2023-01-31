import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
import string
import re
import enchant
from time import time
import itertools
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
mpl.style.use("seaborn")
mpl.rcParams['legend.frameon'] = 'True'
nlp = spacy.load("en_core_web_sm")
en_dict = enchant.Dict('en_US')

# load text files
def preprocess(list_):
    t_prep = []
    for text in list_:
        t = []
        for x in nlp(text):
            x = x.text.replace('-', ' ')
            x = re.sub(r'[^a-zA-Z ]', '', x)
            if x!=' ':
                if len(x)>=3:
                    if en_dict.check(x):
                        t.append(x)
        if t != '':
            if t != ' ':
                t_prep.append(' '.join(t))
    return t_prep
    
def get_text(df):
  """
  input : dataframe
  output: list of sentences from the text files
  """
  df['preprocessed_readme'] = df['readme'].apply(preprocess)
  df = df[df['preprocessed_readme']!='']
  text = df['preprocessed_readme'].values.tolist()
  return df, text

# pre-process data
def lda_preprocess(text, n_features):
  """
  input : list of sentences from the text file
  output: vectorizer, vector repr. of the texts,
          features extracted from the vectorizer
  """

  nlp = spacy.load("en_core_web_sm")

  nostop_text = []
  for sent in text:
    nostop_sentence = []
    doc = nlp(sent)
    for word in doc:
      word = word.lemma_.lower()
      lexeme = nlp.vocab[word]
      if lexeme.is_stop == False:
        nostop_sentence.append(word)
      else:
        continue
        
    nostop_text.append(" ".join(nostop_sentence))

  print(f"Extracting features for LDA . . .")
  tf_vectorizer = TfidfVectorizer(analyzer='word',
                                  max_df=0.95,
                                  min_df=2,
                                  lowercase=True,
                                  max_features=n_features,
                                  token_pattern='[a-zA-Z0-9]{3,}')
  tf = tf_vectorizer.fit_transform(nostop_text)
  tf_feature_names = tf_vectorizer.get_feature_names_out()

  return tf_vectorizer, tf, tf_feature_names


def lda_topic_modeling(vectors, n_components, learning_decay, n_top_words):
  """
  input : vector repr. of the texts, number of topics, 
          learning decay param, top N
  output: lda model
  """

  model = LatentDirichletAllocation(
      n_components=n_components,
      learning_decay=learning_decay,
      random_state=0,
      max_iter=5,
      learning_method="online",
      learning_offset=50.0,
  )
  model.fit(vectors)

  return model


def lda_gridsearch(vectors):
  """
  input : vector repr. of the texts
  output: a plot with the results from the grid-search
  """

  # Define Search Param
  search_params = {
      'n_components': [1, 2, 5, 10, 15, 20, 40],
      'learning_decay': [.1, .2, .5, .7, .9]
  }

  lda_perp = defaultdict(dict)
  lda_like = defaultdict(dict)

  print(f"Starting gridsearch for LDA . . .")
  for lr in search_params['learning_decay']:
    perp, like = [], []
    for comp in search_params['n_components']:
      lda = LatentDirichletAllocation(
          n_components=comp,
          learning_decay=lr,
          random_state=0,
          max_iter=5,
          learning_method="online",
          learning_offset=50.0,
      )
      lda.fit(vectors)
      perp.append(lda.perplexity(vectors))
      like.append(lda.score(vectors))

    lda_perp[lr] = perp
    lda_like[lr] = like

  print(f"Ending gridsearch for LDA . . .")

  lr_decay = search_params['learning_decay']

  plt.figure(figsize=(8, 7))
  mpl.rcParams['legend.frameon'] = 'True'

  for lr in lr_decay[:]:
    plt.plot(search_params['n_components'], lda_like[lr], "--", label=lr)

  plt.title("Choosing Optimal LDA Model", size=15)
  plt.xlabel("Num Topics", size=15)
  plt.ylabel("Log Likelyhood Scores", size=15)
  plt.xticks(size=15)
  plt.yticks(size=15)
  plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
  plt.legend(title='Learning decay',
             loc='best',
             edgecolor='black',
             facecolor='white')

  a = plt.axes([.25, .3, .3, .2], facecolor='w')

  for lr in lr_decay[2:]:
    plt.plot(search_params['n_components'], lda_perp[lr], "--", label=lr)

  plt.xlabel("Num Topics", size=12)
  plt.ylabel("Perplexity Scores", size=12)
  plt.xticks(size=10)
  plt.yticks(size=10)
  plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
  plt.legend(loc='best', edgecolor='black', facecolor='w')
  plt.show()


def major_topic_per_doc(model, vectors, vectorizer, data):
  """
  input : model, vectorizer, vector repr. of the texts
  output: a dataframe with dominant topic per document,
          a dataframe with topic distribution over all documents,
          a dataframe with topic words
  """

  # Create Document - Topic Matrix
  lda_output = model.transform(vectors)
  # print('lda_output: ', lda_output.shape, lda_output)

  # column names
  topicnames = ["Topic-" + str(i) for i in range(model.n_components)]

  # index names
  docnames = ["Doc-" + str(i) for i in range(len(data))]

  # Make the pandas dataframe
  df_document_topic = pd.DataFrame(np.round(lda_output, 2),
                                   columns=topicnames,
                                   index=docnames)

  # Get dominant topic for each document
  dominant_topic = np.argmax(df_document_topic.values, axis=1)
  df_document_topic['dominant_topic'] = dominant_topic

  # Styling
  def color_green(val):
    color = 'blue' if val > .5 else 'black'
    return 'color: {col}'.format(col=color)

  def make_bold(val):
    weight = 700 if val > .5 else 400
    return 'font-weight: {weight}'.format(weight=weight)

  # Apply Style
  df_document_topics = df_document_topic.style.applymap(
      color_green).applymap(make_bold)
  df_topic_distribution = df_document_topic['dominant_topic'].value_counts(
  ).reset_index(name="Num Documents")
  df_topic_distribution.columns = ['Topic Num', 'Num Documents']

  # Topic-Keyword Matrix
  df_topic_keywords = pd.DataFrame(model.components_)

  # Assign Column and Index
  df_topic_keywords.columns = vectorizer.get_feature_names()
  df_topic_keywords.index = [
      "Topic" + str(i) for i in range(model.n_components)
  ]

  return df_document_topics, df_topic_distribution, df_topic_keywords


def plot_top_words(model,
                   feature_names,
                   n_top_words,
                   plot_row_ct,
                   plot_col_ct,
                   figsize=(10, 6)):
  """
  input : model, vectorizer, vector repr. of the texts, 
          features extracted from the vectorizer, top N,
          title of the plot
  output: a plot of top N words from each topic
  """

  fig, axes = plt.subplots(plot_row_ct,
                           plot_col_ct,
                           figsize=figsize,
                           sharex=True)
  axes = axes.flatten()
  for topic_idx, topic in enumerate(model.components_):
    top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
    top_features = [feature_names[i] for i in top_features_ind]
    weights = topic[top_features_ind]

    ax = axes[topic_idx]
    ax.barh(top_features, weights, height=0.7)
    ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 15})
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=14)
    for i in "top right left".split():
      ax.spines[i].set_visible(False)
    fig.suptitle(f"Topics by LDA", fontsize=15)

  plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
  plt.show()


def show_topics(vectorizer, lda_model, n_words=10):
  """
  input : model, vectorizer, number of words,
  output: a dataframe with top N topic words
  """

  keywords = np.array(vectorizer.get_feature_names())
  topic_keywords = []
  for topic_weights in lda_model.components_:
    top_keyword_locs = (-topic_weights).argsort()[:n_words]
    topic_keywords.append(keywords.take(top_keyword_locs))

  # Topic - Keywords Dataframe
  df_topic_keywords = pd.DataFrame(topic_keywords)
  df_topic_keywords.columns = [
      'Word ' + str(i) for i in range(df_topic_keywords.shape[1])
  ]
  df_topic_keywords.index = [
      'Topic ' + str(i) for i in range(df_topic_keywords.shape[0])
  ]

  return df_topic_keywords


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