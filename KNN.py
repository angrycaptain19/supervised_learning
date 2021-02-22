
# Generate data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from util import getSpotifyDataset, getWineDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score 
from matplotlib.colors import ListedColormap
from sklearn.model_selection import validation_curve, learning_curve

def KNN(X, y, train_size, n_neighbors, data_name):
    X = preprocessing.StandardScaler().fit_transform(X)
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size) # 70% training and 30% test
    # #############################################################################
    k_range = range(1,50)
    scores = {}
    scores_list = []
    title = "Accuracy with training size at {}% - {}".format(train_size, data_name)
    for k in k_range:
         #Create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_predict)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))

    plt.plot(k_range, scores_list)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Testing for accuracy")
    plt.show()

    # Obtain scores from learning curve function 
    # cv is the number of folds while performing Cross Validation 
    train_sizes, train_scores, test_scores = learning_curve(KNeighborsClassifier(n_neighbors = n_neighbors), X, y, cv=10, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50)) 
    
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("KNN Learning Curve - {}".format(data_name))
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
