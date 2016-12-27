import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

def plot_decision_regions(X_train, X_test, y_train, y_test, classifier, test_marker=None, resolution=0.02):
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min()-1 , X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1 , X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
        
    # highlight test samples
    if test_marker:
        if isinstance(X_test, pd.core.frame.DataFrame):
            plt.scatter(X_test.ix[:, 0], X_test.ix[:, 1], c='',
                        alpha=1.0, linewidth=1, marker='o',
                        s=55, label='test set')
        else:
            plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                        alpha=1.0, linewidth=1, marker='o',
                        s=55, label='test set')
    
    plt.legend(loc='upper left')
    score(X_test, y_test, classifier)

def score(X_test, y_test, classifier):
    print('Accuracy: {0}'.format(classifier.score(X_test, y_test)))
    y_pred = classifier.predict(X_test)
    print('Number of mislabeled points: {0}'.format((y_test!=y_pred).sum()))