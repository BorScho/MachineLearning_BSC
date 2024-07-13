
# I tried to follow these recommendations: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
# I could not get the BinningClassifier to fullfill all the requirements that are checked by scikits check_estimator-test-suite.
# read more on this here: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator
# The API check is the file binningClassifier_API_chekc.ipynb - the BinningClassifier in this file is the result of trying to implement as
# much as possible of the requirements in the check_estimator test-suite.

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.exceptions import NotFittedError
import numpy as np

class BinningClassifier(BaseEstimator, ClassifierMixin):
    """
    A wrapper around a regression-estimator that converts the output of the regressor into a classification-statement.
    This is done by separating the output-range of the regressor into bins (given by intervals as a user input). 
    The BinningClassifier returns the number/ index of the bin in which the regressor-output lies, starting with 0.
    This wrapper-class around a regressor is necessary to be able to use the regressors in grid-search etc.:
    e.g. from the GridSearchCV documentation:
    [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html]
    class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, ...)
    Parameters:
        estimator: estimator object
        This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.
    """

    """ def __init__(self, regressor=None, intervals=None, **regressor_params):
        self.regressor = regressor
        if regressor_params:
            self.regressor.set_params(**regressor_params)
        self.intervals = intervals """

    def __init__(self, regressor=None, intervals=None):
        self.regressor = regressor
        self.intervals = intervals


    def fit(self, X, y):

        # Validate and check X and y for consistency,deny sparse X, 
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, ensure_2d=True)
        
        # Check if the intervals are provided
        if self.intervals is None or not isinstance(self.intervals, np.ndarray) or not len(self.intervals) >= 2:
            raise ValueError("Intervals must be provided as a numpy array.")
        
        # Check if the regressor is provided
        if self.regressor is None or not isinstance(self.regressor, RegressorMixin):
            raise ValueError("A valid scikit-learn regressor must be provided.")
                
        # Fit the regressor
        self.regressor.fit(X, y)
        
        # Define the array of possible class-attributes:
        self.classes_ = np.arange(len(self.intervals) - 1)
        
        return self


    def predict(self, X):

        # Ensure that the fit was successful
        try:
            check_is_fitted(self, ['regressor', 'intervals', 'classes_'])
        except NotFittedError as exc:
            print("Model is not fitted yet.")

        # Ensure X is a 2d array (i.e. has 2 axises) and denies sparse data
        X = check_array(X, accept_sparse=False, ensure_2d=True)

        # Predict using the regressor
        reg_predictions = self.regressor.predict(X)
        
        # Determine the interval index for each prediction - subtract 1 because np.digitize counts from 1 not 0
        classes = np.digitize(reg_predictions, bins=self.intervals, right=True) - 1
        
        return classes
    

    def predict_proba(self, X):

        # Ensure that the fit was successful
        try:
            check_is_fitted(self, ['regressor', 'intervals', 'classes_'])
        except NotFittedError as exc:
            print("Model is not fitted yet.")
        
        binned_predictions = self.predict(X)
        
        # Create the probability array:
        num_samples = len(binned_predictions)
        num_classes = len(self.classes_)
        proba = np.zeros((num_samples, num_classes))
        
        # Assign probabilities based on class distribution
        for i in range(num_classes):
            #print(f"i: {i} - proba[:,i]: {(binned_predictions.ravel() == i).astype(float)}")
            proba[:, i] = (binned_predictions.ravel() == i).astype(float)
        
        # Normalize probabilities row-wise to sum to 1
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        return proba
    
    def score(self, X, y, sample_weight=None):

        # Ensure that the fit was successful
        try:
            check_is_fitted(self, ['regressor', 'intervals', 'classes_'])
        except NotFittedError as exc:
            print("Model is not fitted yet.")

        # Ensure X is a 2d array (i.e. has 2 axises) and denies sparse data
        X = check_array(X, accept_sparse=False, ensure_2d=True)

        # Validate and check X and y for consistency,deny sparse X, 
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, ensure_2d=True)

        return self.regressor.score(X, y, sample_weight)


    def get_params(self, deep=True):
        return {"regressor": self.regressor, "intervals": self.intervals}
        

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


""" 
# Example usage:
from sklearn.linear_model import LinearRegression
intervals = np.array([10, 20, 30])
clf = BinningClassifier(regressor=LinearRegression(), intervals=intervals)

# Running the common tests
from sklearn.utils.estimator_checks import check_estimator
check_estimator(clf)
"""
