
# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from util import getSpotifyDataset, getWineDataset
from sklearn.metrics import mean_squared_error, accuracy_score 
from sklearn.model_selection import learning_curve
# Create the dataset

def BDT(X, y, train_size, data_name):
    X = StandardScaler().fit_transform(X)
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size) # 70% training and 30% test
    # Fit model
    regr_1 = DecisionTreeClassifier(max_depth=4)

    regr_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
                              n_estimators=300)

    regr_1.fit(X_train, y_train)
    regr_2.fit(X_train, y_train)

    # Predict
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    mse = mean_squared_error(y_test, y_1)
    acc = accuracy_score(y_test, y_2)

    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(AdaBoostClassifier(DecisionTreeClassifier(ccp_alpha=0.00375), n_estimators=300), 
                                                            X, 
                                                            y,
                                                            # Number of folds in cross-validation
                                                            cv=10,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=-1, 
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(0.01, 1.0, 50))

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
    plt.title("BDT Learning Curve - {}".format(data_name))
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()