"""Perception anomaly detection method."""
# -*- coding: utf-8 -*-
# Author: Nassir, Mohammad <nassir.mohammad@airbus.com>
# License: BSD 2 clause

import numpy as np
import math
from scipy.spatial.distance import cdist


class Perception:
    """Perception anomaly detection method.

    ***

    Input data must be standardised if it is multidimensional,
    using for example: sklearn.preprocessing.StandardScaler.

    ***

    Return the anomaly label of each one-dimensional or
    multi-dimensional observation.

    For scores, the greater the score, beyond zero, the more anomalous an
    observation is.

    This function specialises to one-dimensional data when the data only has
    one feature/column.

    The fundamental formula for scoring the unexpectedness of an observation n
    is:

    (S choose n) * 1/(W^{n-1})

    However, this formula is transformed for computational reasons as detailed
    in the accompanying paper and algorithm description to:

    -1/S * (log(S choose n) - (n - 1) * log(W))

    In fact the following is currently used in case of division by zero.
    -1 * (log(S choose n) - (n - 1) * log(W))

    In the multi-dimensional case:

       We take in standardised data and calculate each observations' distance
       from the median using a distance measure, for example Mahalanobis or
       Euclidean. Then, given the one-dimensional data, we apply the
       one-dimensional anomaly detection steps.

    Assumptions on data distribution
    --------------------------------
    The Perception algorithm detects anomalies in one or more dimensions under
    the assumption that:

    'normal' data is found within one majority grouping;

    the anomalies are both rare and relatively far from the main group;

    the feature values are numerical;

    the input is a numpy array: rows correspond to each observation, and
    columns correspond to each feature.

    the data has already been normalised to zero mean and standard
    deviation of 1 for all features.

    Parameters
    ----------
    max_decimal_accuracy : int, optional (default=4)
        The maximum number of decimal places to use for accuracy at the
        one dimensional computation steps

    multi_dim_distance_metric : str, optional, (default=euclidean)
        The distance metric to use between points in multidimensional space.
        e.g., 'euclidean', 'Mahalanobis'.

    multi_dim_method : str, optional (default=DfM)
        The multi-dimensional data method to use. e.g., "DfM", "indpendent".
        These will be implemented in future.

    Attributes
    ----------
    multi_d_medians_ : numpy array of shape (1, n_dimensions)
        The mutli-dimensional medians from training data columns,
        used in predict stage.

    rounding_multiplier_ : integer
        The value to multiply rounded input by (10 ** rounding_multiplier) to
        ensure list of numbers are integers on the same scale, used in predict
        stage.

    training_median_ : integer
        The 1-dimensional data median obtained from training data,
        used at predict.

    S_ : integer
        The total sum of "indicators", i.e. the sum of the input values
        after integerisation using the rounding_multiplier.

    W_ : integer
        The number of integers (or Windows or length of input data).

    scores_ : numpy array
        The anomaly scores of the predicted data. The higher, the more
        abnormal. Anomalies tend to have higher scores. This value is available
        once the detector is fitted.

    labels_ : numpy array of either 0 or 1
        The binary decision labels for each example: 0 normal, 1 anomaly. This
        is obtained (effectively) parameter free.

    anomalies_ : numpy array
        The anomalies from the original input data that are detected by the
        algorithm.

    """

    def __init__(self,
                 max_decimal_accuracy=4,
                 multi_dim_distance_metric='euclidean',
                 multi_dim_method='DfM'):

        self.max_decimal_accuracy = max_decimal_accuracy
        self.multi_dim_distance_metric = multi_dim_distance_metric
        self.multi_dim_method = multi_dim_method

    def fit(self, X):
        """Fit detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        assert type(X) == np.ndarray, "X must by a numpy array"
        assert X.size > 0, "numpy array must be non-empty"
        assert X.ndim < 3, "array can only be 1 or 2 dimensional"

        # get the dimensionality of the data

        # if dimensionality is 1, X.shape[1] not callable
        if X.ndim == 1:
            dimensionality = 1
        elif X.ndim == 2 and X.shape[1] == 1:
            # convert the array to 1-dimensional
            X = np.squeeze(X, axis=1)
            dimensionality = 1
        elif X.shape[1] > 1:
            dimensionality = 2

        if self.multi_dim_method == 'DfM':

            # in the case of multidimensional data, transform data to 1D data
            if dimensionality > 1:

                # calculate the multidimensional median
                self.multi_d_medians_ = np.median(X, axis=0)
                self.multi_d_medians_ = self.multi_d_medians_.reshape(
                    1, np.size(self.multi_d_medians_))

                X = cdist(X,
                          self.multi_d_medians_,
                          self.multi_dim_distance_metric
                          ).ravel()

            self.rounding_multiplier_ = self._get_rounding_multiplier(
                X, self.max_decimal_accuracy)

            # round and multiply to achieve the required decimal accuracy
            # and integer valued numbers
            Xg = self._round_scale(
                X, self.max_decimal_accuracy, self.rounding_multiplier_)

            # find the median of the data (round the median itself in case
            # it is not an integer)
            self.training_median_ = np.round(np.median(Xg))

            # take the distance from the median for all points
            Xf = np.abs(Xg - self.training_median_)

            # take the sum of all the "indicators"
            self.S_ = Xf.sum()

            # get the number of "windows" in the one-dimensional data
            self.W_ = len(Xf)

            return self

    def predict(self, X):
        """Predict if a particular sample is an anomaly or not.

        ----------
        X : numpy array of shape (n_samples, n_features).

        Returns
        -------
        labels_ : numpy array of shape (n_samples,)
            The binary decision labels for each example: 0 normal, 1 anomaly.
        """
        assert type(X) == np.ndarray, "X must by a numpy array"
        assert X.size > 0, "numpy array must be non-empty"

        assert self.S_ >= 0, 'S must be greater than or equal to 0'
        assert self.W_ > 0,'W, number of windows, must be greater than 0'

        assert X.ndim < 3, "array can only be 1 or 2 dimensional"

        # get the dimensionality of the data

        # if dimensionality is 1, X.shape[1] not callable
        if X.ndim == 1:
            dimensionality = 1
        elif X.ndim == 2 and X.shape[1] == 1:
            # convert the array to 1-dimensional
            X = np.squeeze(X, axis=1)
            dimensionality = 1
        elif X.shape[1] > 1:
            dimensionality = 2

        if self.S_ > 0:
            logS_ = math.log(self.S_)
        else:
            logS_ = 0

        # removed (-1/self.S_) in case of division by zero
        sigma = -1
        logW_ = math.log(self.W_)

        if self.multi_dim_method == 'DfM':

            # keep original X for recovering detected anomalies
            if dimensionality > 1:
                Xr = cdist(X,
                           self.multi_d_medians_,
                           self.multi_dim_distance_metric
                           ).ravel()
            else:
                Xr = X

            # round and multiply to get integers on same scale as training data
            Xg = self._round_scale(
                Xr, self.max_decimal_accuracy, self.rounding_multiplier_)

            # self.training_median_ is an integer
            Xf = np.abs(Xg - self.training_median_)

            # S choose 0 == 1, S choose S == 1, hence log(1)=0
            # n>S: function E(C_n) is actually undefined over this range,
            # but will compute to be anomalous

            # using log n! = n*log(n) - n
            # simplified approx of log (S_C_n) using sterling approx gives:
            # S*logS - n*logn - (S-n)*log(S-n)
            with np.errstate(invalid='ignore', divide='ignore'):

                # is n = 0, the approx is 0
                # if S = n, the approx is 0
                # if n > S, the approx must be 0 artificially

                # part A
                SlogS = self.S_*logS_

                # part B
                logXf = np.where(Xf > 0, np.log(Xf), 0)
                XflogXf = Xf * logXf

                # part C
                S_Xf = self.S_ - Xf
                log_S_Xf = np.where(S_Xf > 0, np.log(S_Xf), 0)
                S_XflogS_Xf = S_Xf * log_S_Xf

                # A - B - C
                pre_logS_Choose_Xf = SlogS - XflogXf - S_XflogS_Xf

                # 0 out any values where n>S (i.e. where S_Xf is non-positive)
                mask = np.where(S_Xf < 0, 1, 0)
                logS_Choose_Xf = np.where(mask == 1, 0, pre_logS_Choose_Xf)

            # formula: -1/S * (log(S choose n) - (n - 1) * log(W))
            # removed (-1/self.S_) in case of division by zero
            self.scores_ = sigma*(logS_Choose_Xf - (Xf-1)*logW_)

            # label each score > 1 as anomaly, otherwise leave as 0
            self.labels_ = np.where(self.scores_ > 0, 1, 0)

            # get the anomalies form the original data
            self.anomalies_ = np.array(X[np.where(self.labels_ == 1)])

            return self.labels_

    def fit_predict(self, X):
        """Run fit and predict functions.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels_ : numpy array of shape (n_samples,)
            The binary decision labels for each example: 0 normal, 1 anomaly.
        """
        self.fit(X)
        self.predict(X)

    @staticmethod
    def _get_rounding_multiplier(data, accuracy):
        """Find max number of digits after decimal place of rounded numbers.

        Parameters
        ----------
        data : numpy array of shape (n_samples, n_features)
            The training input data.

        accuracy : integer
            The desired accuracy to round the data to.

        Returns
        -------
        max_exp : integer
            The maximum number of digits after decimal places over all input
            data.
        """
        data_rounded = np.round(data, accuracy)

        data_rounded_str = data_rounded.astype('str')

        max_exp = np.max([el[::-1].find('.') for el in data_rounded_str])

        if max_exp == -1:
            return 0
        else:
            return max_exp

    @staticmethod
    def _round_scale(data, accuracy, rounding_multiplier):
        """Multiply input numbers by rounding multiplier; take integer part.

        Parameters
        ----------
        data : numpy array of shape (n_samples, n_features)
            The training input data.

        accuracy : integer
            The desired accuracy to round the data to.

        rounding_multiplier : integer
            The value to multiply rounded input by (10 ** rounding_multiplier)
            to ensure list of numbers are integers on the same scale, used in
            predict stage.

        Returns
        -------
        integerised_data : numpy array
            The input data integerised (so that every number is an integer).
        """
        data_rounded = np.round(data, accuracy)

        # get rounded raw input as integers
        integerised_data = data_rounded * (10 ** rounding_multiplier)
        integerised_data = integerised_data.astype(int)

        return integerised_data
