from PerceptionAlgorithm import Perception
import numpy as np

data = np.array([2.1, 2.6, 2.4, 2.5, 2.3, 2.1, 2.3, 2.6, 8.2, 8.3])
clf = Perception()
clf.fit(data)
clf.predict(data)
print(clf.anomalies_)
print("Success")

