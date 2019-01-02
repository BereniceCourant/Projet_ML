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
import nltk
import csv
from tqdm import tqdm


def get_features(training_set, indices, node_info, features_TFIDF = None, graph = None, aut_indices = None, graph_citations = None):
    nltk.download('punkt') # for tokenization
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    # we will use three basic features:
    
    # number of overlapping words in title
    overlap_title = []
    
    # temporal distance between the papers
    temp_diff = []
    
    # number of common authors
    auth_0 = []
    auth_1 = []
    auth_2 = []
    
    
    tfidf1 = []
    tfidf2 = []
    tfidf3 = []
    tfidf4 = []
    
    journal = []
    
    dist_citation = []
    
    counter = 0
    for i in range(len(training_set)):
    #for i in tqdm(range(len(training_set))):
        source = training_set[i][0]
        target = training_set[i][1]
        
        index_source = indices[source]
        index_target = indices[target]
        
        source_info = node_info[index_source]
        target_info = node_info[index_target]
        
        # convert to lowercase and tokenize
        source_title = source_info[2].lower().split(" ")
        # remove stopwords
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]
        
        target_title = target_info[2].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]
        
        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")
        (aut_0, aut_1, aut_2) = get_auteurs_012(source_auth, target_auth, aut_indices, graph)
        
        source_journal = source_info[4]
        target_journal = target_info[4]
        
        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        auth_0.append(aut_0)
        auth_1.append(aut_1)
        auth_2.append(aut_2)
        if (source_journal == target_journal):
            journal.append(1)
        else:
            journal.append(0)
        
        if (features_TFIDF != None):
            source_tfidf = features_TFIDF[index_source]
            target_tfidf = features_TFIDF[index_target]
            tfidf1.append((source_tfidf.dot(target_tfidf.T))[0, 0]/np.sqrt(source_tfidf.dot(source_tfidf.T)[0,0]*target_tfidf.dot(target_tfidf.T)[0, 0]))
        
        
        if (graph_citations != None):
            s = source_info[0]
            t = target_info[0]
            d = graph_citations.shortest_paths_dijkstra(s, t)[0][0]
            if d >= 20:
                d = 20
            dist_citation.append(d)
            
      
        counter += 1
        if counter % 1000 == True:
            print( counter, "training examples processsed")
            pass
    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    if (features_TFIDF == None):
        training_features = np.array([overlap_title, temp_diff, auth_0, journal]).T
    else:
        if (graph_citations == None):
            training_features = np.array([overlap_title, temp_diff, auth_0, auth_1, auth_2, journal, tfidf1]).T
        else:
            training_features = np.array([overlap_title, temp_diff, auth_0, auth_1, auth_2, journal, tfidf1, dist_citation]).T

    
    # scale
    training_features = preprocessing.scale(training_features)
    
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
            graph.add_vertex(k)
        
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
    if graph != None:
        for a1 in auteurs1:
            k1 = aut_indices[a1]
            neighbors1 = graph.neighborhood(k1)
            neighbors2 = graph.neighborhood(k1, order=2)
            aut_1 += len(set(neighbors1).intersection(set(auteurs2_indices)))
            aut_2 += len((set(neighbors2).difference(set(neighbors1))).intersection(set(auteurs2_indices)))
    aut_0 = len(set(auteurs1).intersection(set(auteurs2)))
    return (aut_0, aut_1, aut_2)
    
def construct_graph_citations(training_set, predictions, indices, node_info):
    graph = igraph.Graph(directed=False)
    articles_vu = set()
    
    for i in range(len(training_set)):
        source = training_set[i][0]
        target = training_set[i][1]
        
        index_source = indices[source]
        index_target = indices[target]
        
        source_info = node_info[index_source]
        target_info = node_info[index_target]
        
        id_source = source_info[0]
        id_target = target_info[0]
        
        if not (id_source in articles_vu):
            articles_vu.add(id_source)
            graph.add_vertex(id_source)
        if not (id_target in articles_vu):
            articles_vu.add(id_target)
            graph.add_vertex(id_target)
        
        if predictions[i] == 1:
            graph.add_edge(id_source, id_target)
        
    return graph
    
    

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
            
            

def split_training_set(training_set, per, validation = True):
    to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*per)))
    training_set_reduced = [training_set[i] for i in to_keep]
    if validation:
        validation_set = [training_set[i] for i in range(len(training_set)) if not i in to_keep]
    else:
        validation_set = []
    return training_set_reduced, validation_set
    
    

def open_set(file_name):
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        set  = list(reader)

    set = [element[0].split(" ") for element in set]
    
    return set
    
    
def init():
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
    
    return testing_set, training_set, indices, features_TFIDF, node_info


def multiple_split(training_set, n, prop):
    print("split set...", end = '')
    validation = training_set[:30000]
    training = training_set[30000:]
    training_sets = []
    for i in range(n):
        t = split_training_set(training, prop, validation = False)[0]
        training_sets.append(t)
    print("done")
    return validation, training_sets

def result_svm(prop):
    testing_set, training_set, indices, features_TFIDF, node_info = init()
    
    # randomly select 5% of training set
    print("reduce set...", end = '')
    training_set_red = split_training_set(training_set, prop, validation = False)[0]
    training_set_reduced, validation_set = split_training_set(training_set_red, 0.7)
    print("done")
    
    #Create Graph
    print("construct graph...", end = '')
    (graph, aut_indices) = construct_graph_auteurs(indices, node_info)
    print("done")
    
    #Compute features for training set
    training_features = get_features(training_set_reduced, indices, node_info, features_TFIDF, graph, aut_indices)
    
    # convert labels into integers then into column array
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    # initialize basic SVM
    #classifier = svm.LinearSVC(max_iter=10000, C=1e-1)
    classifier = svm.SVC(max_iter=1000000, C = 1e1)
    #classifier = svm.SVC(max_iter=100000, kernel='sigmoid', coef0 = 0, gamma = 'scale')
    #classifier = svm.SVC(max_iter=100000, kernel='poly', coef0 = 10, gamma = 'scale')

    
    # train
    classifier.fit(training_features, labels_array)
    
    # test
    # we need to compute the features for the testing set
    validation_features = get_features(validation_set, indices, node_info, features_TFIDF, graph, aut_indices)
    
    # issue predictions
    predictions_SVM = list(classifier.predict(validation_features))
    acc = get_accuracy(validation_set, predictions_SVM)
    
    
    #get_Result_for submit
    testing_features = get_features(testing_set, indices, node_info, features_TFIDF, graph, aut_indices)
    predictions_SVM_submit = list(classifier.predict(testing_features))
    write_pred(predictions_SVM_submit, name = "predictions_svm_"+str(prop)+".csv")
    
    return acc
    

def logistic_regression(prop):
    testing_set, training_set, indices, features_TFIDF, node_info = init()
    
    # randomly select 5% of training set
    print("reduce set...", end = '')
    training_set_red = split_training_set(training_set, prop, validation = False)[0]
    training_set_reduced, validation_set = split_training_set(training_set_red, 0.7)
    print("done")
    
    #Compute features for training set
    training_features = get_features(training_set_reduced, indices, node_info, features_TFIDF)
    
    # convert labels into integers then into column array
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    # initialize basic SVM
    classifier = LogisticRegression(max_iter=10000)
    
    # train
    classifier.fit(training_features, labels_array)
    
    # test
    # we need to compute the features for the testing set
    validation_features = get_features(validation_set, indices, node_info, features_TFIDF)
    
    # issue predictions
    predictions_SVM = list(classifier.predict(validation_features))
    acc = get_accuracy(validation_set, predictions_SVM)
    
    
    # #get_Result_for submit
    # testing_features = get_features(testing_set, indices, node_info, features_TFIDF)
    # predictions_SVM_submit = list(classifier.predict(testing_features))
    # write_pred(predictions_SVM_submit, name = "predictions_svm_"+str(prop)+".csv")
    
    return acc
    

def get_predictions(training_set, validation_set, indices, node_info, features_TFIDF):
    #Compute features for training set
    training_features = get_features(training_set, indices, node_info, features_TFIDF)
    
    # convert labels into integers then into column array
    labels = [int(element[2]) for element in training_set]
    labels = list(labels)
    labels_array = np.array(labels)
    
    # initialize basic SVM
    #classifier = svm.LinearSVC(max_iter=10000)
    classifier = svm.SVC(max_iter=10000, C=1e1)
    # train
    classifier.fit(training_features, labels_array)
    # test
    # we need to compute the features for the testing set
    validation_features = get_features(validation_set, indices, node_info, features_TFIDF)
    
    # issue predictions
    predictions_SVM = list(classifier.predict(validation_features))
    
    return predictions_SVM
    

def bagging(n, prop):
    testing_set, training_set, indices, features_TFIDF, node_info = init()

    validation, trainings_set = multiple_split(training_set, n, prop)
    all_predictions = []
    for i in range(n):
        print("test" + str(i))
        training = trainings_set[i]
        predictions = get_predictions(training, validation, indices, node_info, features_TFIDF)
        all_predictions.append(predictions)
    
    final_prediction = []
    m = len(validation)
    for k in range(m):
        res = [0, 0]
        for i in range(n):
            p = all_predictions[i][k]
            res[p] += 1
        if res[0] > res[1]:
            final_prediction.append(0)
        else:
            final_prediction.append(1)
    
    acc = get_accuracy(validation, final_prediction)
    
    # #get result for submit
    # all_predictions_test = []
    # for i in range(n):
    #     print("test" + str(i))
    #     training = trainings_set[i]
    #     predictions = get_predictions(training, testing_set, indices, node_info, features_TFIDF)
    #     all_predictions_test.append(predictions)
    
    # final_prediction_test = []
    # m = len(testing_set)
    # for k in range(m):
    #     res = [0, 0]
    #     for i in range(n):
    #         p = all_predictions_test[i][k]
    #         res[p] += 1
    #     if res[0] > res[1]:
    #         final_prediction_test.append(0)
    #     else:
    #         final_prediction_test.append(1)
    # predictions_SVM_submit = final_prediction_test
    # write_pred(predictions_SVM_submit, name = "predictions_bagging_"+str(n)+"_" + str(prop) + ".csv")

    return acc
    

def random_forest(n, prop):
    testing_set, training_set, indices, features_TFIDF, node_info = init()
    validation = training_set[:30000]
    training = training_set[30000:]
    training_set_reduced = split_training_set(training, prop, validation = False)[0]
    features = get_features(training_set_reduced, indices, node_info, features_TFIDF)
    
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    clf = RandomForestClassifier(n_estimators = n)
    clf.fit(features, labels_array)
    
    features_validation = get_features(validation, indices, node_info, features_TFIDF)
    predictions = clf.predict(features_validation)
    
    acc = get_accuracy(validation, predictions)
    
    #get result for submit
    testing_features = get_features(testing_set, indices, node_info, features_TFIDF)
    predictions_SVM_submit = list(clf.predict(testing_features))
    write_pred(predictions_SVM_submit, name = "predictions_randomForest_"+str(n)+ "_" + str(prop)+".csv")
    
    return acc
    
    
def adaBoosting(n, lr, prop):
    testing_set, training_set, indices, features_TFIDF, node_info = init()
    validation = training_set[:30000]
    training = training_set[30000:]
    training_set_reduced = split_training_set(training, prop, validation = False)[0]
    features = get_features(training_set_reduced, indices, node_info, features_TFIDF)
    
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    #classifier = svm.SVC(max_iter=100000, kernel='poly', coef0 = 10, gamma = 'scale')
    classifier = RandomForestClassifier(n_estimators = 100)
    ada_boost = AdaBoostClassifier(classifier, n_estimators=n,learning_rate=lr, algorithm='SAMME')
    ada_boost.fit(features, labels_array)
    
    features_validation = get_features(validation, indices, node_info, features_TFIDF)
    predictions = ada_boost.predict(features_validation)
    
    acc = get_accuracy(validation, predictions)
    
    # #get result for submit
    # testing_features = get_features(testing_set, indices, node_info, features_TFIDF)
    # predictions_SVM_submit = list(ada_boost.predict(testing_features))
    # write_pred(predictions_SVM_submit, name = "predictions_randomForest_"+str(n)+ "_" + str(prop)+".csv")
    
    return acc
    

def gradientBoosting(n, lr, prop):
    testing_set, training_set, indices, features_TFIDF, node_info = init()
    print("reduce set...", end = '')
    training_set_red = split_training_set(training_set, prop, validation = False)[0]
    training_set_reduced, validation = split_training_set(training_set_red, 0.7)
    print("done")
    print("construct graph...", end = '')
    (graph, aut_indices) = construct_graph_auteurs(indices, node_info)
    print("done")
    features = get_features(training_set_reduced, indices, node_info, features_TFIDF, graph, aut_indices)
    
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    classifier = svm.SVC(max_iter=100000, kernel='poly', coef0 = 10, gamma = 'scale')
    grad_boost = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=1, random_state=0)
    grad_boost.fit(features, labels_array)
    
    
    
    features_validation = get_features(validation, indices, node_info, features_TFIDF, graph, aut_indices)
    predictions = grad_boost.predict(features_validation)
    
    acc = get_accuracy(validation, predictions)
    
    predictions_training = grad_boost.predict(features)
    acc_training = get_accuracy(training_set_reduced, predictions_training)
    print("precision training = " + str(acc_training))
    
    #get result for submit
    testing_features = get_features(testing_set, indices, node_info, features_TFIDF, graph, aut_indices)
    predictions_SVM_submit = list(clf.predict(testing_features))
    write_pred(predictions_SVM_submit, name = "predictions_gradientBoosting_"+ "svc_poly_coef0="+ str(10)+ "_ n="+str(n)+ "_prop=" + str(prop)+".csv")
    
    return acc


def svm_dist_citation(prop):
    testing_set, training_set, indices, features_TFIDF, node_info = init()
    print("reduce set...", end = '')
    training_set_red = split_training_set(training_set, prop, validation = False)[0]
    training_set_reduced, validation = split_training_set(training_set_red, 0.7)
    print("done")
    print("construct graph...", end = '')
    (graph, aut_indices) = construct_graph_auteurs(indices, node_info)
    print("done")
    
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    classifier_1 = svm.SVC(max_iter=100000, C=1e1)
    
    features_train_1 = get_features(training_set_reduced, indices, node_info, features_TFIDF, graph, aut_indices)
    classifier_1.fit(features_train_1, labels_array)
    
    predictions_train_1 = classifier_1.predict(features_train_1)
    print("construct graph...", end = '')
    graph_train = construct_graph_citations(training_set_reduced, predictions_train_1, indices, node_info)
    print("done")
    
    features_train_2 = get_features(training_set_reduced, indices, node_info, features_TFIDF, graph, aut_indices, graph_train)
    classifier_2 = svm.SVC(max_iter=1000000, C=1e1)
    classifier_2.fit(features_train_2, labels_array)
    
    predictions_train_2 = classifier_2.predict(features_train_2)

    
    features_test_1 = get_features(validation, indices, node_info, features_TFIDF, graph, aut_indices)
    predictions_test_1 = classifier_1.predict(features_test_1)
    print("construct graph...", end = '')
    graph_test = construct_graph_citations(validation, predictions_test_1, indices, node_info)
    print("done")
    
    features_test_2 = get_features(validation, indices, node_info, features_TFIDF, graph, aut_indices, graph_test)
    predictions_test_2 = classifier_2.predict(features_test_2)
    
    
    acc = get_accuracy(validation, predictions_test_2)
    acc_train_1 = get_accuracy(training_set_reduced, predictions_train_1)
    acc_train_2 = get_accuracy(training_set_reduced, predictions_train_2)
    acc_test_1 = get_accuracy(validation, predictions_test_1)
    
    print("acc train 1 = " + str(acc_train_1))
    print("acc train 2 = "+str(acc_train_2))
    print("acc test 1 = "+ str(acc_test_1))
    
    return acc


    
    
    
    

###################
# random baseline #
###################

#random_predictions = np.random.choice([0, 1], size=len(testing_set))
#write_pred(random_predictions)
        
# note: Kaggle requires that you add "ID" and "category" column headers

###############################
# beating the random baseline #
###############################

# the following script gets an F1 score of approximately 0.66

# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes



## the following shows how to construct a graph with igraph
## even though in this baseline we don't use it
## look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

#edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

## some nodes may not be connected to any other node
## hence the need to create the nodes of the graph from node_info.csv,
## not just from the edge list

#nodes = IDs

## create empty directed graph
#g = igraph.Graph(directed=True)
 
## add vertices
#g.add_vertices(nodes)
 
## add edges
#g.add_edges(edges)

# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set




