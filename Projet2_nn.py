import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse.linalg import svds
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import gensim.downloader as api
from gensim.test.utils import get_tmpfile
from gensim.models import word2vec
import gensim
import nltk
import csv
from tqdm import tqdm


def get_features(training_set, indices, node_info, features_TFIDF, graph, aut_indices, TFIDF_titre, graph_citation=None, articles_indices=None):
    nltk.download('punkt') # for tokenization
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    # we will use three basic features:
    
    # number of overlapping words in title
    title_s = []
    title_t = []
    
    # temporal distance between the papers
    temp_diff = []
    
    # number of common authors
    auth_0 = []
    auth_1 = []
    
    
    tfidf_s = []
    tfidf_t = []
    
    journal = []
    
    dist_citation = []
    nbr_article_per_author_s = [[], [], [], [], []]
    nbr_article_per_author_t = [[], [], [], [], []]
    author_articles = set_articles_from_author(node_info)

    
    counter = 0
    for i in range(len(training_set)):
    #for i in tqdm(range(len(training_set))):
        source = training_set[i][0]
        target = training_set[i][1]
        
        index_source = indices[source]
        index_target = indices[target]
        
        source_info = node_info[index_source]
        target_info = node_info[index_target]
        
        source_id = source_info[0]
        target_id = target_info[0]
        
        source_tfidf_title = TFIDF_titre[index_source]
        target_tfidf_title = TFIDF_titre[index_target]
        title_s.append(source_tfidf_title)
        title_t.append(target_tfidf_title)
        
        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")
        (aut_0, aut_1, aut_2) = get_auteurs_012(source_auth, target_auth, aut_indices, graph)
        
        source_journal = source_info[4]
        target_journal = target_info[4]
        
        source_abstract = source_info[5].lower().split(" ")
        source_abstract = [token for token in source_abstract if token not in stpwds]
        source_abstract = [stemmer.stem(token) for token in source_abstract]
        
        target_abstract = target_info[5].lower().split(" ")
        target_abstract = [token for token in target_abstract if token not in stpwds]
        target_abstract = [stemmer.stem(token) for token in target_abstract]
        
        
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        auth_0.append(aut_0)
        auth_1.append(aut_1)
        
        if (source_journal == target_journal):
            journal.append(1)
        else:
            journal.append(0)
            
        source_tfidf = features_TFIDF[index_source]
        target_tfidf = features_TFIDF[index_target]
        tfidf_s.append(source_tfidf)
        tfidf_t.append(target_tfidf)
        
        if graph_citation is not None:
            dist = get_distance_citation(source_id, target_id, graph_citation, articles_indices)
            dist_citation.append(dist)
            
            L1, L2 = get_nbr_articles_per_author(index_source, index_target, indices, node_info, graph, articles_indices, author_articles)
            for k in range(5):
                nbr_article_per_author_s[k].append(L1[k])
                nbr_article_per_author_t[k].append(L2[k])
            
            
            
      
        counter += 1
        if counter % 1000 == True:
            print( counter, "training examples processsed")
            pass
            
    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)

    L = [temp_diff, auth_0, auth_1, journal]
    
    n_t = len(title_s)
    m_t = len(title_s[0])
    m2_t = len(title_t[0])
    if m_t!=m2_t:
        print("error"+ str(m_t)+" " + str(m2_t))
    T = [[] for k in range(2*m_t)]
    for i in range(n_t):
        for j in range(m_t):
            T[j].append(title_s[i][j])
            T[j+m_t].append(title_t[i][j])
    for X in T:
        L.append(X)
        
    n = len(tfidf_s)
    m = len(tfidf_s[0])
    m2 = len(tfidf_t[0])
    if m!=m2:
        print("error"+ str(m)+" " + str(m2))
    T = [[] for k in range(2*m)]
    for i in range(n):
        for j in range(m):
            T[j].append(tfidf_s[i][j])
            T[j+m].append(tfidf_t[i][j])
    for X in T:
        L.append(X)
    
    if graph_citation is not None:
        L.append(dist_citation)
        for k in range(5):
            L.append(nbr_article_per_author_s[k])
            L.append(nbr_article_per_author_t[k])
            
    training_features = np.array(L).T
    # scale
    training_features = preprocessing.scale(training_features)
    
    print(np.shape(training_features))
        
    return training_features
    


def construct_graph_auteurs(indices, node_info):
    n_aut = 0
    aut_vu = dict()
    graph = igraph.Graph(directed=False)
    edges = []

        
    for article in node_info:
        auteurs = article[3].split(',')
        for a in auteurs:
            if a in aut_vu:
                k = aut_vu[a]
            else:
                k = n_aut
                n_aut += 1
                aut_vu[a] = k
            graph.add_vertex(name=k)
        
        for i in range(len(auteurs)):
            for j in range(i+1, len(auteurs)):
                k1 = aut_vu[auteurs[i]]
                k2 = aut_vu[auteurs[j]]
                edges.append((k1, k2))
    
    print("n_auteurs = "+str(n_aut)+ " ", end = '')
    
    graph.add_edges(edges)

    return graph, aut_vu

def get_auteurs_012(auteurs1, auteurs2, aut_indices, graph):
    aut_0 = 0
    aut_1 = 0
    aut_2 = 0
    auteurs1_indices = [aut_indices[a] for a in auteurs1]
    auteurs2_indices = [aut_indices[a] for a in auteurs2]
    if not graph is None:
        for a1 in auteurs1:
            k1 = aut_indices[a1]
            neighbors1 = graph.neighborhood(k1)
            neighbors2 = graph.neighborhood(k1, order=2)
            aut_1 += len(set(neighbors1).intersection(set(auteurs2_indices)))
            aut_2 += len((set(neighbors2).difference(set(neighbors1))).intersection(set(auteurs2_indices)))
    aut_0 = len(set(auteurs1).intersection(set(auteurs2)))
    return (aut_0, aut_1, aut_2)

def construct_graph_citations(training_set, indices, node_info):
    graph = igraph.Graph(directed=False)
    articles_vu = dict()
    
    k = 0
    for info in node_info:
        id = info[0]
        graph.add_vertex(k)
        articles_vu[id] = k
        k += 1

    
    #for i in range(len(training_set)):
    for i in tqdm(range(len(training_set))):
        source = training_set[i][0]
        target = training_set[i][1]
        
        index_source = indices[source]
        index_target = indices[target]
        
        source_info = node_info[index_source]
        target_info = node_info[index_target]
        
        id_source = articles_vu[source_info[0]]
        id_target = articles_vu[target_info[0]]
        
        graph.add_edge(id_source, id_target)
        
    return graph, articles_vu

def get_distance_citation(article1, article2, graph, articles_indices):
    id1 = articles_indices[article1]
    id2 = articles_indices[article2]
    connected = graph.are_connected(id1, id2)
    if connected:
        graph.delete_edges([(id1, id2)])
    dist = graph.shortest_paths_dijkstra(id1, id2)[0][0]
    if connected:
        graph.add_edge(id1, id2)
    if dist >= 5:
        dist = 5
    return dist

def set_articles_from_author(node_info):
    author_articles = dict()
    n = len(node_info)
    for i in range(n):
        info = node_info[i]
        id = info[0]
        authors = info[3].split(",")
        for auth in authors:
            if auth in author_articles:
                author_articles[auth].append(id)
            else:
                author_articles[auth] = [id]
    return author_articles
    

def get_nbr_articles_per_author(source_index, target_index, indices, node_info, graph, articles_indices, author_articles):
    id1 = articles_indices[node_info[source_index][0]]
    id2 = articles_indices[node_info[target_index][0]]
    connected = graph.are_connected(id1, id2)
    if connected:
        graph.delete_edges([(id1, id2)])
    authors1 = node_info[source_index][3].split(",")
    L1 = []
    for auth in authors1:
        s = 0
        articles = author_articles[auth]
        for ar in articles:
            s += len(graph.neighborhood(articles_indices[ar]))
        L1.append(s)
    L1.sort(reverse=True)
    for k in range(5-len(L1)):
        L1.append(0)
    L1 = L1[:5]
    
    authors2 = node_info[target_index][3].split(",")
    L2 = []
    for auth in authors2:
        s = 0
        articles = author_articles[auth]
        for ar in articles:
            s += len(graph.neighborhood(articles_indices[ar]))
        L2.append(s)
    L2.sort(reverse=True)
    for k in range(5-len(L2)):
        L2.append(0)
    L2 = L2[:5]
    
    if connected:
        graph.add_edge(id1, id2)
    
    return L1, L2
    
    
    

def get_accuracy(testing_set, prediction):
    tp = 0
    fn = 0
    fp = 0
    n = len(testing_set)
    for i in range(n):
        pred = prediction[i]
        res = int(testing_set[i][2])
        if (pred == 1 and res == 0):
            fp += 1
        if (pred == 1 and res == 1):
            tp += 1
        if (pred == 0 and res == 1):
            fn += 1
    p = float(tp) / float(tp+fp)
    r = float(tp) / float(tp + fn)
    return 2*p*r/(p+r)
    
    
    
def write_pred(prediction, name = "predictions.csv"):
    # write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
    predictions_SVM = [[str(i), str(prediction[i])] for i in range(len(prediction))]
    
    with open(name,"w") as pred1:
        csv_out = csv.writer(pred1)
        csv_out.writerow(["id", "category"])
        csv_out.writerows(predictions_SVM)
            
    
def init(n_svd=100, n_svd_title=20):
    testing_set = open_set("testing_set.txt")
    training_set = open_set("training_set.txt")
    with open("node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info  = list(reader)
    
    IDs = [element[0] for element in node_info]
    indices = dict()
    for (i, j) in enumerate(IDs):
        indices[j] = i
    
    # compute TFIDF vector of each paper
    print("tfidf...", end = '')
    corpus = [element[5] for element in node_info]
    vectorizer = TfidfVectorizer(stop_words="english")
    # each row is a node in the order of node_info
    features_TFIDF = vectorizer.fit_transform(corpus)
    print("done")
    print("svd...", end='')
    u, s, v = svds(features_TFIDF, k=n_svd, return_singular_vectors="u")
    features_TFIDF_reduced = u.dot(np.diag(s))
    print("done")
    #tfidf titre
    titres = [element[2] for element in node_info]
    vectorizer_titre = TfidfVectorizer(stop_words="english")
    features_TFIDF_title = vectorizer_titre.fit_transform(titres)
    u_titres, s_titres, v_titres = svds(features_TFIDF_title, k=n_svd_title, return_singular_vectors="u")
    features_TFIDF_title = u_titres.dot(np.diag(s_titres))

    return testing_set, training_set, indices, features_TFIDF_reduced, features_TFIDF_title, node_info

def neural_network(prop, alpha=1e-5, layers=(150, 100, 50), n_svd=128, n_svd_title=20, compute = True):
    testing_set, training_set, indices, features_TFIDF, features_TFIDF_titre, node_info = init(n_svd, n_svd_title)
    print("reduce set...", end = '')
    training_set_red = training_set[30000:]
    validation = training_set[:30000]
    training_set_reduced = training_set_red[:int(len(training_set_red)*prop)]#split_training_set(training_set, prop, validation = False)[0]
    print("done")
    print("construct graph...", end = '')
    (graph, aut_indices) = construct_graph_auteurs(indices, node_info)
    print("done")
    
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    if compute:
        training_features = get_features(training_set_reduced, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre)
        np.save("training_features", training_features)
    else:
        print("load features...", end='')
        training_features = np.load("training_features_title_graph.npy")#[:int(len(training_set_red)*prop)]
        print("done")
        
    clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=layers)
    clf.fit(training_features, labels_array)
    
    if compute:
        validation_features = get_features(validation, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre)
        np.save("validation_features", validation_features)
    else:
        print("load features..", end='')
        validation_features = np.load("validation_features_title_graph.npy")
        print("done")
    predictions = clf.predict(validation_features)
    acc = get_accuracy(validation, predictions)
    
    predictions_train = clf.predict(training_features)
    acc_train = get_accuracy(training_set_reduced, predictions_train)
    print("train = "+str(acc_train))
    
    
    #get result for submit
    if compute:
        testing_features = get_features(testing_set, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre)
        np.save("testing_features", testing_features)
    else:
        testing_features = np.load("testing_features_title_graph.npy")
    predictions_SVM_submit = list(clf.predict(testing_features))
    write_pred(predictions_SVM_submit, name = "predictions_nn_graph"+ "alpha="+str(alpha)+ "_prop=" + str(prop)+".csv")
    
    return acc

    

def save_all(n_svd, n_svd_title, prop, first=True):
    testing_set, training_set, indices, features_TFIDF, features_TFIDF_titre, node_info = init(n_svd, n_svd_title)
    print("construct graph...", end = '')
    (graph, aut_indices) = construct_graph_auteurs(indices, node_info)
    print("done")
    
    training_set_reduced = training_set[30000:]
    training_set_red = training_set_reduced[:int(len(training_set_reduced)*prop)]
    validation_set = training_set[:30000]
    print("construct graph citation...", end='')
    if first:
        (graph_citation, articles_indices) = construct_graph_citations(training_set_red, indices, node_info)
        graph_citation.save("graph_citation", format="pickle")
    else:
        graph_citation = igraph.Graph.Read("graph_citation", format="pickle")
    print("done")
        
    training_features = get_features(training_set_red, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_citation=graph_citation, articles_indices=articles_indices)
    np.save("training_features_title_graph2", training_features)
    validation_features = get_features(validation_set, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_citation=graph_citation, articles_indices=articles_indices)
    np.save("validation_features_title_graph2", validation_features)
    testing_features = get_features(testing_set, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_citation=graph_citation, articles_indices=articles_indices)
    np.save("testing_features_title_graph2", testing_features)