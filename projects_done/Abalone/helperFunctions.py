# helper functions Abalone_2.ipynb:
import numpy as np


def binIndex(x:float, bin_edges:np.array)->int:
    """
        ! OBSOLETE: is implemented in numpy as np.digitize !
        binIndex: gives the index of the bin in the bin_edge-array that contains x
        Input:
            x:float
            bin_edges: np.array of float - defining the borders of a connected set of left-open intervalls: [a,b,c,...] <=> ]a,b], ]b, c], ]c,...] etc.
        Output:
            out: int - the index of the intervall in which the input x is contained.
            out: -1 if x is not contained in any of the intervalls.
    """

    for i in range(len(bin_edges) - 1):
        if bin_edges[i + 1] >= x and x > bin_edges[i]:
            return int(i)
    return -1


def binIndexVec(X:np.array, bin_edges:np.array)->np.array:
    """
        ! OBSOLETE: is implemented in numpy as np.digitize !
        binIndexVec:
        gives a list of the indeces of the bins in the bin_edge-array that contain x for any value x in the array X.
        Output:
            out: an array of indeces of the bins containing the elements of X
    """
    result = np.zeros(len(X), dtype=int)
    for i, x in enumerate(X):
        result[i] = binIndex(x, bin_edges)
    return result


def bin_scorer(y_true:np.array, y_pred:np.array, bin_edges:np.array)->np.array:
    """
        bin_scorer:
        transforms each value of the y_pred-array into the bin-index of the bin containing it and calculates the accuracy of these
        transformed array compared with y_true. 
        This is a score_func for use in make_scorer, see:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer

        Input:
            y_true: reference values, ground truth
            y_pred: values that have to be "binned" and then compared to y_true
            bin_edges: array defining the borders of the bins
        
        Output:
            accuracy_score of y_true vs binned y_pred

    """
    from sklearn.metrics import accuracy_score
    y_pred_binned = binIndexVec(y_pred, bin_edges)
    return accuracy_score(y_true, y_pred_binned).mean()


def spot_shot_classifiers(classifiers, X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array)->np.array:
    """
        spot_shot_classifiers:
        Trains an array of classifiers on X_train, y_train and calculates their accuracies on X_test, y_test.
        Uses StandardScaler on the input data.

        Input:
            classifiers: an array of scikit-learn classifiers
            X_train: training input data
            y_train: training labels for X_train
            X_test: test input data
            y_test: test labesl for X_test
        
        Output:
            list of accuracy_scores of y_test and predictions on X_test
    """
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    accuracies = {}
    for clf_name, clf_instance in classifiers.items():
        classifier = Pipeline([
            ("ssc", StandardScaler()), # standard scaler because of "sag"-solver in logistic-regression
            (clf_name, clf_instance)
        ])

        classifier.fit(X_train, y_train)

        # print prediction :
        y_pred = classifier.predict(X_test)
        accuracies[clf_name] = accuracy_score(y_true=y_test, y_pred=y_pred)
    return accuracies


def spot_shot_classifiers_cross_val(validation_folds, classifiers, X_train:np.array, y_train:np.array)->np.array:
    """
        spot_shot_classifiers:
        Trains an array of classifiers on X_train, y_train and calculates their accuracies on X_test, y_test.
        Uses StandardScaler on the input data.

        Input:
            validation_folds: number of cross-validation folds, defaults to 5
            classifiers: an array of scikit-learn classifiers
            X_train: training input data
            y_train: training labels for X_train
        
        Output:
            dictionary of accuracy mean scores for each classifier
    """
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    accuracies = {}

    if validation_folds == None:
        validation_folds = 5

    for clf_name, clf_instance in classifiers.items():
        classifier = Pipeline([
            ("ssc", StandardScaler()), # standard scaler because of "sag"-solver in logistic-regression
            (clf_name, clf_instance)
        ])

        accuracy = cross_val_score(classifier, X_train, y_train, scoring= "accuracy", cv = validation_folds)
        accuracies[clf_name] = accuracy.mean()

    return accuracies


def spot_shot_regressors(regressors, bin_edges:np.array, X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array)->np.array:
    """
        spot_shot_regressors:
        Trains an array of regressors on X_train, y_train and calculates their accuracies on X_test, y_test. 
        Puts the predictions made on X_test into bins and compares y_test to those binned predictions. 
        This is done in order to be able to use regressors as classifiers

        Input:
            regressors: an array of scikit-learn regressors
            bin_edges: array defining the borders of the bins
            X_train: training input data
            y_train: training labels for X_train
            X_test: test input data
            y_test: test labesl for X_test
        
        Output:
            list of accuracy_scores of y_test and predictions on X_test
    """
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    accuracies = {}
    for reg_name, reg_instance in regressors.items():
        regressor = Pipeline([
            ("ssc", StandardScaler()), 
            (reg_name, reg_instance)
        ])

        regressor.fit(X_train, y_train)

        # print prediction accuracy:
        y_pred = regressor.predict(X_test)
        y_pred_discrete = binIndexVec(y_pred, bin_edges)
        accuracies[reg_name] = accuracy_score(y_true=y_test, y_pred=y_pred_discrete)
    return accuracies


"""
Example of usage and test for the helper functions:

# Example / test of the function "binIndex":

print('Example / test of the function "binIndex"')

age_bins = np.arange(0.5, 30, step=1)
x = 10.2

print(f"x: {x}")
print(f"Edges :{age_bins}")

#bi = binIndex(x, age_bins)
bi = hf.binIndex(x, age_bins)
if bi > -1:
    print(f"Containing bin edges of {x} are: [{age_bins[bi]}, {age_bins[bi + 1]}]")
elif bi == -1:
    print(f"Return value: {bi} ,i.e. x not within binned range")



# Example / test of the function "binIndexVec":

print('Example / test of the function "binIndexVec"')

age_bins = np.arange(0.5, 30, step=1)
X = np.array([1,2,3,55,12])

print(f"X: {X}")
print(f"Edges :{age_bins}")

#bi_vec = binIndexVec(X, age_bins)
bi_vec = hf.binIndexVec(X, age_bins)

print(f"binIndexVec results: {bi_vec}")

for i, x in enumerate(X):
    bi = bi_vec[i]
    if bi > -1:
        print(f"Containing bin edges of {x} are: [{age_bins[bi]}, {age_bins[bi + 1]}]")
    elif bi == -1:
        print(f"Return value: {bi} ,i.e. {x} not within binned range")
"""

def plot_confusion_matrix_sns(y_true, y_pred, labels=None, fig_size=(10,10)):
    """
        plot_confusion_matrix_sns:
        uses Seaborn to plot a confusion matrix.

        Input:
            y_true: array-like of shape (n_samples,)
                    Ground truth (correct) target values.

            y_pred: array-like of shape (n_samples,)
                    Estimated targets as returned by a classifier.

            labels: array-like of shape (n_classes), default=None
                    List of labels to index the matrix. This may be used to reorder 
                    or select a subset of labels. 
                    If None is given, those that appear at least once in y_true or y_pred are used in sorted order.
            fig_size: default=(10,10), the plot-size
            
        Output:
            returns the scikit confusion-matrix.
        
        Usage Example:
            cm = plot_confusion_matrix_sns(y_true=y_test, y_pred=y_pred, labels=clf.classes_)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    plt.figure(figsize=fig_size)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SVC confusion matrix")
    plt.show()

    return cm
