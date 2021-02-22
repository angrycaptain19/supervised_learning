# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

def DT(X, y, train_size, data_name):
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
        # Fit classification model
        dt = DecisionTreeClassifier ()
        path = dt.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")
        
        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)    
        print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
              clfs[-1].tree_.node_count, ccp_alphas[-1]))

        # %%
        # For the remainder of this example, we remove the last element in
        # ``clfs`` and ``ccp_alphas``, because it is the trivial tree with only one
        # node. Here we show that the number of nodes and tree depth decreases as alpha
        # increases.
        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title("Depth vs alpha")
        fig.tight_layout()

        # %%
        # Accuracy vs alpha for training and testing sets
        # ----------------------------------------------------
        # When ``ccp_alpha`` is set to zero and keeping the other default parameters
        # of :class:`DecisionTreeClassifier`, the tree overfits, leading to
        # a 100% training accuracy and 88% testing accuracy. As alpha increases, more
        # of the tree is pruned, thus creating a decision tree that generalizes better.
        # In this example, setting ``ccp_alpha=0.015`` maximizes the testing accuracy.
        train_scores = [clf.score(X_train, y_train) for clf in clfs]
        test_scores = [clf.score(X_test, y_test) for clf in clfs]

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas, train_scores, marker='o', label="train",
                drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker='o', label="test",
                drawstyle="steps-post")
        ax.legend()
        plt.show()
        # %%
        best_alpha = 0.040790348647614105
        # %%
        # Create CV training and test scores for various training set sizes
        train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(ccp_alpha=best_alpha), 
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


        print(train_scores)
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
        plt.title("DT Learning Curve - {}".format(data_name))
        plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
        # %%
