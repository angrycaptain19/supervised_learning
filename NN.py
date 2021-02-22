import numpy as np
import warnings
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import learning_curve

def NN(X, y, train_size, data_name):
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=500)
    with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
    # mlp.fit(X_train, y_train)

    # y_pred_train = mlp.predict(X_train)
    # y_pred_test = mlp.predict(X_test)
    # print("--------------- TRAIN RESULTS --------------------")
    # print(confusion_matrix(y_train, y_pred_train))
    # print(classification_report(y_train, y_pred_train))
    # print("--------------- TEST RESULTS--------------------")
    # print(confusion_matrix(y_test, y_pred_test))
    # print(classification_report(y_test, y_pred_test))

    train_sizes, train_scores, test_scores = learning_curve(mlp, 
                                                            X, 
                                                            y,
                                                            # Number of folds in cross-validation
                                                            cv=5,
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
    plt.title("NN Learning Curve - {}".format(data_name))
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()