# Author: Mohammad Nassir <nassir.mohammad@airbus.com>
# License: BSD 2 clause

import numpy as np
import scipy
from decimal import Decimal
from math import log, pi

class Perception:

    """

    Perception anomaly detection algorithm.

    Return the anomaly label and associated score of each observation. The greater the score, the more anomalous.

    This function specialises to one-dimensional data when the data only has one feature.

    The fundamental formula for scoring the unexpectedness of an observation is:

    (S choose n) * 1/(W^{n-1})

    However, this formula is transformed for computational reasons as detailed in the accompanying paper and code.

    In the multi-dimensional case:

       We take in scaled data and calculate each observations' distance from the median using a
       distance measure, i.e. Mahalanobis or Euclidean. Then, given the one-dimensional data, we apply the
       one-dimensional anomaly detection algorithm as before.



    Assumptions on data distribution
    --------------------------------
    The Perception algorithm detects anomalies in one or more dimensions under the assumption that the 'normal' data is
    found within one majority grouping;

    that the anomalies are both rare and relatively far from the main group;

    that the data values are continuous numbers;

    that input is a numpy array: rows correspond to each observation, and columns correspond to each feature.

    that the data has already been normalised to zero mean and standard deviation of 1 for all features.


    Parameters (can be left at default values; users need not be concerned about setting them)
    ----------

    max_decimal_accuracy : int, optional (default=4)
        The maximum number of decimal places to use for accuracy in single dimensional data.

    multi_dim_distance_metric : str, optional, (default=euclidean)
        The distance metric to use between points in multidimensional space.

    multi_dim_method : str, optional (default=DfM)
        The multi-dimensional data method to use.



    Attributes
    ----------

    """

    def __init__(self,
                 max_decimal_accuracy=4,
                 multi_dim_distance_metric='euclidean',
                 multi_dim_method='DfM'):

        # initialise maximum decimal accuracy of input data
        self.max_decimal_accuracy_ = max_decimal_accuracy

        # initialise the distance metric to use
        self.multi_dim_distance_metric_ = multi_dim_distance_metric  # 'euclidean' # euclidean euclidean Mahalanobis

        # initialise the type of multi dimensional anomaly detection algorithm - "DfM", "indpendent"
        self.multi_dim_method_ = multi_dim_method

        # store the mutli-dimensional medians from training data columns, used for predict
        self.multi_d_medians_ = None

        # store the median from the training data to be used at predict. For 1D data.
        self.training_median_ = None

        # value to multiply rounded input by to ensure list of numbers are integers on the same scale
        self.rounding_multiplier_ = None

        # initialise total sum of indicators S to zero
        self.S_ = 0

        # initialise the number of integers (or Windows) to zero (length of input)
        self.W_ = 0

        # create a zero initialised array to hold each example score, any score > 0 is an anomaly
        self.scores_ = None

        # create holder for labels, 0 normal, 1 anomaly for each example.
        self.labels_ = None

        # holder for raw anomaly values
        self.anomalies_ = None

        # holds the train and test distances used as input to the algorithm
        self.distances_train_ = None
        self.distances_test_ = None

    def fit(self, X):

        input_data = None

        assert type(X) == np.ndarray, "X must by a numpy array"
        assert X.ndim > 0, "X must have dimension greater than or equal to 1"

        if self.multi_dim_method_ == 'DfM':

            # in the case of multidimensional data, transform the data to 1D data
            if X.ndim > 1:
                # calculate the multidimensional median
                self.multi_d_medians_ = np.median(X, axis=0)
                self.multi_d_medians_ = self.multi_d_medians_.reshape(1, np.size(self.multi_d_medians_))

                # pick the appropriate distance metric.
                input_data = scipy.spatial.distance.cdist(X,
                                                 self.multi_d_medians_,
                                                 self.multi_dim_distance_metric_
                                                 ).ravel()

            # round and multiply to achieve the required decimal accuracy
            if input_data is None:
                X_r = self._round(X, self.max_decimal_accuracy_)
            else:
                X_r = self._round(input_data, self.max_decimal_accuracy_)

            # find the median of the data (round the median itself in case it is a decimal value)
            self.training_median_ = np.round(np.median(X_r))

            # take absolute values of the points to make them all postiive and the median is now zero
            X_d = np.abs(X_r - self.training_median_)

            # take the sum of all the indicators
            self.S_ = X_d.sum()

            # get the length of the number of integers to ascertain how many "windows" we have
            self.W_ = len(X_d)

    def predict(self, X):

        input_data = None

        assert type(X) == np.ndarray, "X must by a numpy array"
        assert X.ndim > 0, "X must have dimension greater than or equal to 1"

        if self.multi_dim_method_ == 'DfM':

            if X.ndim > 1:
                input_data = scipy.spatial.distance.cdist(X,
                                                 self.multi_d_medians_,
                                                 self.multi_dim_distance_metric_
                                                 ).ravel()

            # round to integers to achieve the required accuracy using rounding multipler from training stage,
            # keep original X unchanged
            if input_data is None:
                X_r = np.round(X, self.rounding_multiplier_)
            else:
                X_r = np.round(input_data, self.rounding_multiplier_)

            # multiply to obtain integers on same scale as training data
            X_g = X_r * (10 ** self.rounding_multiplier_)

            X_g = np.abs(X_g - self.training_median_) # self.training_median_ is an integer

            # vectorise the calculation method to apply it to each integer
            vectorised_f = np.vectorize(self._compute_window_gas)

            # apply the function
            self.scores_ = vectorised_f(X_g)

            # label each score > 1 as anomaly, otherwise leave as 0
            self.labels_ = np.where(self.scores_ > 0, 1, 0)

            # get the anomalies in the original data decimal places (not the rounded data)
            self.anomalies_ = np.array(X[np.where(self.labels_ == 1)])

            # sort the anomalies
            self.anomalies_ = np.sort(self.anomalies_)

            return self.labels_

    def fit_predict(self, X):

        self.fit(X)
        self.predict(X)

    # round every number to required decimal accuracy. Then find maximum number of decimals
    # in set of numbers (this may be less than the provided decimal accuracy), so that
    # we can multiply every number by correct power to get whole integers.
    # TODO: can this be made more efficient or not required at all?
    def _round(self, X, accuracy):

        # round to the required accuracy
        X_rounded = np.round(X, accuracy)

        # convert all the numbers to string
        X_rounded_str = X_rounded.astype('str')

        # map each number to decimal to have an iterator
        X_rounded_dec = map(Decimal, X_rounded_str)

        # use list comprehension to get number of exponents in number, and take maximum
        # this can be a maximum the same as accuracy

        self.rounding_multiplier_ = max([abs(el.as_tuple().exponent) for el in X_rounded_dec])

        # get rounded raw input as integers
        vals = X_rounded * (10 ** self.rounding_multiplier_)

        # return values as integers
        return vals.astype(int)

    # calculate the gas value, which is the negative log of Exp_C_w divided by S
    def _compute_window_gas(self, n):

        value = -1 / self.S_ * (self._log_binomial(n) - (n - 1) * log(self.W_))

        return value

    # calculate and return the log of the binomial coefficient C_S_n
    def _log_binomial(self, n):

        # S choose 0 == 1, S choose S == 1, hence log(1)
        # n>S: function E(C_n) is actually undefined over this range, but will compute to be anomalous
        if n == 0 or n == self.S_ or n > self.S_:
            return 0

        return (self._stirling_log_approx(self.S_) -
                self._stirling_log_approx(n) -
                self._stirling_log_approx(self.S_ - n))

    # efficient method to return approximate log(n!)
    def _stirling_log_approx(self, n):
        return n * log(n) - n + (0.5 * log(n)) + (0.5 * log(2 * pi))


