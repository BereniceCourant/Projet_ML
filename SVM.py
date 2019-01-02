import random
import numpy as np
#import igraph
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


def get_features(training_set, indices, node_info, features_TFIDF = None):
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
    comm_auth = []
    
    tfidf1 = []
    tfidf2 = []
    tfidf3 = []
    tfidf4 = []
    
    journal = []
    
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
        
        source_journal = source_info[4]
        target_journal = target_info[4]
        
        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
        if (source_journal == target_journal):
            journal.append(1)
        else:
            journal.append(0)
        
        if (features_TFIDF != None):
            source_tfidf = features_TFIDF[index_source]
            target_tfidf = features_TFIDF[index_target]
            tfidf1.append((source_tfidf.dot(target_tfidf.T))[0, 0]/np.sqrt(source_tfidf.dot(source_tfidf.T)[0,0]*target_tfidf.dot(target_tfidf.T)[0, 0]))
            # m1 = np.mean(source_tfidf)
            # m2 = np.mean(source_tfidf)
            # product = source_tfidf.dot(target_tfidf.T)[0, 0] - np.sum(source_tfidf)*m2-np.sum(target_tfidf)*m1+m1*m2*source_tfidf.shape[1]
            # norm_source = source_tfidf.dot(source_tfidf.T)[0, 0] - 2*np.sum(source_tfidf)*m1+m1*m1*source_tfidf.shape[1]
            # norm_target = target_tfidf.dot(target_tfidf.T)[0, 0] - 2*np.sum(target_tfidf)*m2+m2*m2*source_tfidf.shape[1]
            # tfidf2.append(product/np.sqrt(norm_source*norm_target))
            # dif = source_tfidf - target_tfidf
            # tfidf3.append(np.sqrt((dif.dot(dif.T))[0, 0]))
            # tfidf4.append((source_tfidf.dot(target_tfidf.T))[0, 0]/(np.sqrt(source_tfidf.dot(source_tfidf.T)[0,0]*target_tfidf.dot(target_tfidf.T)[0, 0]) - (source_tfidf.dot(target_tfidf.T))[0, 0]))


            
        
        counter += 1
        if counter % 1000 == True:
            print( counter, "training examples processsed")
            pass
    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    if (features_TFIDF == None):
        training_features = np.array([overlap_title, temp_diff, comm_auth, journal]).T
    else:
        training_features = np.array([overlap_title, temp_diff, comm_auth, journal, tfidf1]).T
    
    # scale
    training_features = preprocessing.scale(training_features)
    
    return training_features
    
    

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
    
    #Compute features for training set
    training_features = get_features(training_set_reduced, indices, node_info, features_TFIDF)
    
    # convert labels into integers then into column array
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    # initialize basic SVM
    #classifier = svm.LinearSVC(max_iter=10000, C=1e-1)
    #classifier = svm.SVC(max_iter=100000, C = 1e1)
    #classifier = svm.SVC(max_iter=100000, kernel='sigmoid', coef0 = 0, gamma = 'scale')
    classifier = svm.SVC(max_iter=100000, kernel='poly', coef0 = 5, gamma = 'scale')

    
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
    
    classifier = svm.SVC(max_iter=100000)
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
    validation = training_set[:30000]
    training = training_set[30000:]
    training_set_reduced = split_training_set(training, prop, validation = False)[0]
    features = get_features(training_set_reduced, indices, node_info, features_TFIDF)
    
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    
    classifier = svm.SVC(max_iter=10000)
    grad_boost = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=1, random_state=0)
    grad_boost.fit(features, labels_array)
    
    features_validation = get_features(validation, indices, node_info, features_TFIDF)
    predictions = grad_boost.predict(features_validation)
    
    acc = get_accuracy(validation, predictions)
    
    # #get result for submit
    # testing_features = get_features(testing_set, indices, node_info, features_TFIDF)
    # predictions_SVM_submit = list(clf.predict(testing_features))
    # write_pred(predictions_SVM_submit, name = "predictions_randomForest_"+str(n)+ "_" + str(prop)+".csv")
    
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




