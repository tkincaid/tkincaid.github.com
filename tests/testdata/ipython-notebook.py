import pandas as pd
import numpy as np

class CustomModel(object):

  def fit(self, X, Y):
    '''This is where you would fit your model to the data

    Parameters
    ----------
    X : pandas.DataFrame
        Contains your data - you can think of it as simply loaded from
        pandas.read_csv, but any transformed or derived features you
        have included come along
    Y : pandas.Series
        Has as many elements as X has rows - these are the predictions
        that go along with the data in X

    Returns
    -------
    self : CustomModel
        We utilize operator chaining and need to be able to run
        ``self.fit().predict()`` or similar.
    '''
    return self

  def predict(self, X):
    '''This is the prediction method that you would call

    The output is a numpy ndarray, having a single column and as many
    rows as X

    Parameters
    ----------
    X : pandas.DataFrame
        The data on which to make a prediction using your newly fit model

    Returns
    -------
    Y : numpy.ndarray
        With a single column, and the same number of rows as ``X``
    '''
    return np.ones((len(X), 1))