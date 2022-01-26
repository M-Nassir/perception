Perception Anomaly Detection.

This project contains the prototypical implemenation of the algorithm described in the paper "Anomaly Detection using Principle of Human Perception" 
that can be found at: https://arxiv.org/abs/2103.12323

The perception algorithm detects global anomalies in univariate and multivariate data parameter-free! Just give it the standardised data as a numpy array and it will return
the label (normal or anomalous) of each observation. The algorithm also gives each observation a score that can be useful for further analysis and investigation.

Notes on the algorithm:

  Returns the anomaly label and associated score of each observation.

  The greater the score, above zero, the more anomalous an observation is.

  This function specialises to one-dimensional data when the data only has one feature.

  The fundamental formula for scoring the unexpectedness of an observation is:
  (S choose n) * 1/(W^{n-1})

  However, this formula is transformed for computational reasons as detailed in the accompanying paper and code to:
  -1/S * (log(S choose n) - (n - 1) * log(W))

In the multi-dimensional case:

   We take in scaled data and calculate each observations' distance from the median using a
   distance measure, for example Mahalanobis or Euclidean. Then, given the one-dimensional data, we apply the
   one-dimensional anomaly detection algorithm as before.
