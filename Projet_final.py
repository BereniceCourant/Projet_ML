import numpy as np
import igraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from scipy.sparse.linalg import svds
from sklearn.neural_network import MLPClassifier
import nltk
import csv


    
def init(n_svd=100, n_svd_title=20):
    '''Initialized some variables: testing_set, training_set, node_info, indices, and tfidf vectors for titles and abstracts'''
    testing_set = open_set("testing_set.txt")
    training_set = open_set("training_set.txt")
    with open("node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info  = list(reader)
    
    #Create the indices dictionary which gives for each article id its rank in the node_info list
    IDs = [element[0] for element in node_info]
    indices = dict()
    for (i, j) in enumerate(IDs):
        indices[j] = i
    
    #Compute TFIDF vector of each abstract
    print("tfidf...", end = '')
    corpus = [element[5] for element in node_info]
    vectorizer = TfidfVectorizer(stop_words="english")
    features_TFIDF = vectorizer.fit_transform(corpus)
    print("done")
    #Reduce the dimension of those tfidf vectors with an SVD
    print("svd...", end='')
    u, s, v = svds(features_TFIDF, k=n_svd, return_singular_vectors="u")
    features_TFIDF_reduced = u.dot(np.diag(s))
    print("done")
    #Do the same for the titles
    titles = [element[2] for element in node_info]
    vectorizer_title = TfidfVectorizer(stop_words="english")
    features_TFIDF_title = vectorizer_title.fit_transform(titles)
    u_titles, s_titles, v_titles = svds(features_TFIDF_title, k=n_svd_title, return_singular_vectors="u")
    features_TFIDF_title = u_titles.dot(np.diag(s_titles))

    return testing_set, training_set, indices, features_TFIDF_reduced, features_TFIDF_title, node_info
    

def init_set():
    '''Only initialise the testing_set and training_set. Used when the features have already been computed and saved in a file'''
    testing_set = open_set("testing_set.txt")
    training_set = open_set("training_set.txt")
    return testing_set, training_set
    

def get_features(training_set, indices, node_info, features_TFIDF, graph, aut_indices, TFIDF_titre, graph_citation, articles_indices):
    '''Compute all the features for the given set'''
    nltk.download('punkt') # for tokenization
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    
    # tfidf vectors for source and target titles
    title_s = []
    title_t = []
    
    # temporal distance between the papers
    temp_diff = []
    
    # number of common authors and number of authors who already collaborate with at least one of the author
    auth_0 = []
    auth_1 = []
    
    #tfidf vectors for source and target abstracts
    tfidf_s = []
    tfidf_t = []
    
    #similarity of the journals
    journal = []
    
    #distance between the two articles in the graph of citations
    dist_citation = []
    
    #5 maximum numbers of articles that have been quoted by the source and target authors
    nbr_article_per_author_s = [[], [], [], [], []]
    nbr_article_per_author_t = [[], [], [], [], []]
    author_articles = set_articles_from_author(node_info)

    
    counter = 0
    for i in range(len(training_set)):
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
        (aut_0, aut_1) = get_authors_01(source_auth, target_auth, aut_indices, graph)
        
        source_journal = source_info[4]
        target_journal = target_info[4]
        
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
        
        dist = get_are_articles_connected_5(source_id, target_id, graph_citation, articles_indices)
        dist_citation.append(dist)
            
        L1, L2 = get_nbr_articles_per_author(index_source, index_target, indices, node_info, graph_citation, articles_indices, author_articles)
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
    
    #transpose the list of tfidf vectors in order to have the good format for the final result features
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
    
    L.append(dist_citation)
    print(dist_citation)
    for k in range(5):
        L.append(nbr_article_per_author_s[k])
        L.append(nbr_article_per_author_t[k])
            
    training_features = np.array(L).T
    
    # scale
    training_features = preprocessing.scale(training_features)
            
    return training_features
    


def construct_graph_author(indices, node_info):
    '''Construct the graph of collaboration between authors: there is an edge between authors a1 and a2 if a1 and a2 have already collaborate once.'''
    n_aut = 0 
    aut_seen = dict() #convert author name into an indice
    graph = igraph.Graph(directed=False)
    edges = []

        
    for article in node_info:
        authors = article[3].split(',')
        #Add the vertices that doesn't already exist
        for a in authors:
            if a in aut_seen:
                k = aut_seen[a]
            else:
                k = n_aut
                n_aut += 1
                aut_seen[a] = k
            graph.add_vertex(k)
        
        #Add the corresponding edges
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                k1 = aut_vu[authors[i]]
                k2 = aut_vu[authors[j]]
                edges.append((k1, k2))
        
    graph.add_edges(edges)

    return graph, aut_seen

def get_authors_01(authors1, authors2, aut_indices, graph):
    '''Get the authors at distance 0 or 1 in the previous graph'''
    aut_0 = 0
    aut_1 = 0
    authors1_indices = [aut_indices[a] for a in authors1]
    authors2_indices = [aut_indices[a] for a in authors2]
    for a1 in auteurs1:
        k1 = aut_indices[a1]
        neighbors1 = graph.neighborhood(k1)
        aut_1 += len(set(neighbors1).intersection(set(authors2_indices)))
    aut_0 = len(set(authors1).intersection(set(authors2)))
    return (aut_0, aut_1)


def construct_graph_references(training_set, indices, node_info):
    '''Constuct the graph of references: there is an edge between articles ar1 and ar2 if ar1 quotes ar2 or ar2 quotes ar1'''
    articles_seen = dict() #transforme article into indices
    Neighbors = []
    
    #add all the vertices
    k = 0
    for info in node_info:
        id = info[0]
        Neighbors.append([])
        articles_seen[id] = k
        k += 1
    
    #add all the edges
    for i in range(len(training_set)):
        source = training_set[i][0]
        target = training_set[i][1]
        
        index_source = indices[source]
        index_target = indices[target]
        
        source_info = node_info[index_source]
        target_info = node_info[index_target]
        
        id_source = articles_seen[source_info[0]]
        id_target = articles_seen[target_info[0]]
        
        if int(training_set[i][2]) == 1:
            Neighbors[id_source].append(id_target)
            Neighbors[id_target].append(id_source)
    
    return Neighbors, articles_seen

def get_are_articles_connected_5(article1, article2, graph, articles_indices):
    '''Get the distance between two articles in the previous graph where we remove the direct edge between article1 and article2 if it exists. If the distance is greater than 5, return -1'''
    id1 = articles_indices[article1]
    id2 = articles_indices[article2]
    L = [id1]
    Seen = [False]*len(graph)
    Seen[id1] = True
    ite = 0
    k = 0
    finish = False
    first = (id2 in graph[id1]) #to remove the direct edge betweend id1 and id2
    while ite < 5 and not finish:
        n = len(L)
        while k < n:
            s = L[k]
            if s==id2:
                finish=True
                break
            else:
                Neighbors = graph[s]
                for v in Neighbors:
                    if v==id2 and first:#remove the edge
                        first=False
                    else:
                        if not Seen[v]:
                            Seen[v] = True
                            L.append(v)
            k += 1
            
        if not finish:
            ite += 1

    if finish:
        return ite
    else:
        return -1


def set_articles_from_author(node_info):
    '''Get all the articles written by every authors'''
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
    

def get_nbr_articles_per_author(source_index, target_index, indices, node_info, Neighbors, articles_indices, author_articles):
    '''Get the number of articles quoted by every author in a given article. Then return the 5 greater numbers for each source and target articles'''
    id1 = articles_indices[node_info[source_index][0]]
    id2 = articles_indices[node_info[target_index][0]]
    
    #result for the source article
    authors1 = node_info[source_index][3].split(",")
    L1 = []
    for auth in authors1:
        s = 0
        articles = author_articles[auth]
        for ar in articles:
            id_article = articles_indices[ar]
            s += len(Neighbors[id_article]) - 1
        L1.append(s)
    L1.sort(reverse=True)
    for k in range(5-len(L1)):
        L1.append(0)
    L1 = L1[:5]
    
    #result for the target article
    authors2 = node_info[target_index][3].split(",")
    L2 = []
    for auth in authors2:
        s = 0
        articles = author_articles[auth]
        for ar in articles:
            id_article = articles_indices[ar]
            s += len(Neighbors[id_article]) - 1
        L2.append(s)
    L2.sort(reverse=True)
    for k in range(5-len(L2)):
        L2.append(0)
    L2 = L2[:5]
    
    return L1, L2
    

def get_accuracy(testing_set, prediction):
    '''Compute the accuracy of the given prediction'''
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
    '''write the result of the given prediction in a csv file'''
    #Add the column name
    predictions_SVM = [[str(i), str(prediction[i])] for i in range(len(prediction))]
    
    with open(name,"w") as pred1:
        csv_out = csv.writer(pred1)
        csv_out.writerow(["id", "category"])
        csv_out.writerows(predictions_SVM)
            

def neural_network(prop, alpha=1e-5, layers=(150, 100, 50), n_svd=128, n_svd_title=20, compute = True):
    '''Create a neural network, train it on a given proportion (prop) of the training set, compute the accuracy on the validation set and compute the result on the testing set. If compute is True we compute all the features, else we only need to load them from a file.'''
    if compute:
        testing_set, training_set, indices, features_TFIDF, features_TFIDF_titre, node_info = init(n_svd, n_svd_title)
    else:
        testing_set, training_set = init_set()
    print("reduce set...", end = '')
    training_set_red = training_set[30000:]
    validation = (training_set[:30000])[:int(30000*prop)]
    training_set_reduced = training_set_red[:int(len(training_set_red)*prop)]
    print("done")
    if compute:
        print("construct graph...", end = '')
        (graph, aut_indices) = construct_graph_author(indices, node_info)
        (graph_references, articles_indices) = construct_graph_references(training_set, indices, node_info)
        print("done")
    
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    if compute:
        training_features = get_features(training_set_reduced, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_references, articles_indices)
        np.save("training_features_graph_title_"+str(prop), training_features)
    else:
        print("load features...", end='')
        training_features = np.load("training_features_graph_title_1.npy")#[:int(len(training_set_red)*prop)]
        print("done")
    
    #Create and train the model
    clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=layers)
    clf.fit(training_features, labels_array)
    
    if compute:
        validation_features = get_features(validation, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_references, articles_indices)
        np.save("validation_features_graph_title_"+str(prop), validation_features)
    else:
        print("load features..", end='')
        validation_features = np.load("validation_features_graph_title_1.npy")
        print("done")
      
    #Get the result on the validation set
    predictions = clf.predict(validation_features)
    acc = get_accuracy(validation, predictions)
    
    
    #Print the result of a prediction on the training set in order to be sure that the neural network is not overfitting
    predictions_train = clf.predict(training_features)
    acc_train = get_accuracy(training_set_reduced, predictions_train)
    print("train = "+str(acc_train))
    
    
    #Compute the result of the testing set for submission
    if compute:
        testing_features = get_features(testing_set, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_references, articles_indices)
        np.save("testing_features_graph_title_"+str(prop), testing_features)
    else:
        testing_features = np.load("testing_features_graph_title_1.npy")
        
    predictions_SVM_submit = list(clf.predict(testing_features))
    name_layers = ""
    for x in layers:
        name_layers += int(x) + "_"
    write_pred(predictions_SVM_submit, name = "predictions_nn_graph"+ "alpha="+str(alpha)+ "_layers="+name_layers+"_prop=" + str(prop)+".csv")
    
    return acc
    

def save_all(n_svd, n_svd_title, prop, first=True):
    '''Compute all the features and save them in a file'''
    testing_set, training_set, indices, features_TFIDF, features_TFIDF_titre, node_info = init(n_svd, n_svd_title)
    print("construct graph...", end = '')
    (graph, aut_indices) = construct_graph_author(indices, node_info)
    print("done")
    
    training_set_reduced = training_set[30000:]
    training_set_red = training_set_reduced[:int(len(training_set_reduced)*prop)]
    validation_set = training_set[:30000]
    print("construct graph citation...", end='')
    (graph_references, articles_indices) = construct_graph_references(training_set_red, indices, node_info)
    print("done")
        
    training_features = get_features(training_set_red, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_references, articles_indices)
    np.save("training_features_title_graph_"+str(prop), training_features)
    validation_features = get_features(validation_set, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_references, articles_indices)
    np.save("validation_features_title_graph_"+str(prop), validation_features)
    testing_features = get_features(testing_set, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_references, articles_indices)
    np.save("testing_features_title_graph_"+str(prop), testing_features)


def final_submission(alpha=1e-5, layers=(150, 100, 50, 20)):
    '''Function to be used to test our model on the final testing set. It used the features of the training previously saved in a file and the compute the new features of the testing set.'''
    n_svd = 128
    n_svd_title = 20
    
    testing_set, training_set, indices, features_TFIDF, features_TFIDF_titre, node_info = init(n_svd, n_svd_title)
    
    print("construct graphs...", end='')
    (graph, aut_indices) = construct_graph_author(indices, node_info)
    (graph_references, articles_indices) = construct_graph_references(training_set, indices, node_info)
    print("done")
    
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
 
    print("load features...", end='')
    training_features = np.load("all_training_features.npy")
    print("done")
        
    clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=layers)
    clf.fit(training_features, labels_array)
    
    predictions_train = clf.predict(training_features)
    acc_train = get_accuracy(training_set_reduced, predictions_train)
    print("accuracy train = "+str(acc_train))
    
    #get result for submit
    testing_features = get_features(testing_set, indices, node_info, features_TFIDF, graph, aut_indices, features_TFIDF_titre, graph_references, articles_indices)
    predictions_SVM_submit = list(clf.predict(testing_features))
    write_pred(predictions_SVM_submit, name = "result_prediction.csv")