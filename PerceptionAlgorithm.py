# Author: Mohammad Nassir <nassir.mohammad@airbus.com>
# License: BSD 2 clause

import numpy as np
import scipy
from decimal import Decimal
from math import log, pi


class Perception:

    """

    Perception anomaly detection algorithm.

    Return the anomaly label and associated score of each observation.

    The greater the score, beyond zero, the more anomalous an observation is.

    This function specialises to one-dimensional data when the data only has one feature.

    The fundamental formula for scoring the unexpectedness of an observation is:

    (S choose n) * 1/(W^{n-1})

    However, this formula is transformed for computational reasons as detailed in the accompanying paper and code to:

    -1/S * (log(S choose n) - (n - 1) * log(W))

    In the multi-dimensional case:

       We take in scaled data and calculate each observations' distance from the median using a
       distance measure, for example Mahalanobis or Euclidean. Then, given the one-dimensional data, we apply the
       one-dimensional anomaly detection algorithm as before.



    Assumptions on data distribution
    --------------------------------
    The Perception algorithm detects anomalies in one or more dimensions under the assumption that the 'normal' data is
    found within one majority grouping;

    that the anomalies are both rare and relatively far from the main group;

    that the feature values are numerical;

    that input is a numpy array: rows correspond to each observation, and columns correspond to each feature.

    that the data has already been normalised to zero mean and standard deviation of 1 for all features.


    Parameters (left at default values; users need not be concerned about setting them except in special circumstances)
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

        assert type(X) == np.ndarray, "X must by a numpy array"
        assert X.ndim > 0, "X must have dimension greater than or equal to 1"

        if self.multi_dim_method_ == 'DfM':

            # in the case of multidimensional data, transform the data to 1D data,
            if X.ndim > 1:
                # calculate the multidimensional median
                self.multi_d_medians_ = np.median(X, axis=0)
                self.multi_d_medians_ = self.multi_d_medians_.reshape(1, np.size(self.multi_d_medians_))

                # pick the appropriate distance metric.
                X = scipy.spatial.distance.cdist(X,
                                                 self.multi_d_medians_,
                                                 self.multi_dim_distance_metric_
                                                 ).ravel()

            # get rounding multiplier
            self.rounding_multiplier_ = self._get_rounding_multiplier(X, self.max_decimal_accuracy_)

            # round and multiply to achieve the required decimal accuracy and integer valued numbers
            X_g = self._round_scale(X, self.max_decimal_accuracy_, self.rounding_multiplier_)

            # find the median of the data (round the median itself in case it is a decimal value)
            self.training_median_ = np.round(np.median(X_g))

            # take the distance from the median for all points
            X_f = np.abs(X_g - self.training_median_)

            # take the sum of all the indicators
            self.S_ = X_f.sum()

            # get the length of the number of integers to ascertain how many "windows" we have
            self.W_ = len(X_f)

    def predict(self, X):

        assert type(X) == np.ndarray, "X must by a numpy array"
        assert X.ndim > 0, "X must have dimension greater than or equal to 1"

        if self.multi_dim_method_ == 'DfM':

            # keep original X for later
            if X.ndim > 1:
                X_r = scipy.spatial.distance.cdist(X,
                                                 self.multi_d_medians_,
                                                 self.multi_dim_distance_metric_
                                                 ).ravel()
            else:
                X_r = X

            # round and multiply to obtain integers on same scale as training data
            X_g = self._round_scale(X_r, self.max_decimal_accuracy_, self.rounding_multiplier_)

            X_f = np.abs(X_g - self.training_median_) # self.training_median_ is an integer

            # vectorise the calculation method to apply it to each integer
            vectorised_f = np.vectorize(self._compute_window_gas)

            # apply the function
            self.scores_ = vectorised_f(X_f)

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

    # Round every number to required decimal accuracy. Then find maximum number of decimals
    # in set of numbers (this may be less than the provided decimal accuracy), so that
    # we can multiply every number by correct power to get whole integers.
    # TODO: can this be made more efficient or not required at all?

    @staticmethod
    def _get_rounding_multiplier(data, accuracy):

        # round to the required accuracy
        data_rounded = np.round(data, accuracy)

        # convert all the numbers to string
        data_rounded_str = data_rounded.astype('str')

        # map each number to decimal to have an iterator
        data_rounded_dec = map(Decimal, data_rounded_str)

        # use list comprehension to get number of exponents in number, and take maximum
        return max([abs(el.as_tuple().exponent) for el in data_rounded_dec])

    # Multiply every number by the rounding multiplier and take only the integer part for required accuracy
    @staticmethod
    def _round_scale(data, accuracy, rounding_multiplier):

        # round to the required accuracy
        data_rounded = np.round(data, accuracy)

        # get rounded raw input as integers
        vals = data_rounded * (10 ** rounding_multiplier)

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
    @staticmethod
    def _stirling_log_approx(n):
        return n * log(n) - n + (0.5 * log(n)) + (0.5 * log(2 * pi))


