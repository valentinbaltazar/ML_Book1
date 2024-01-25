from distutils.log import Log
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(list(iris.keys()))

X = iris["data"][:,3:] #petal width

y  = (iris["target"]==2).astype(np.int) # 1 if Iris virginica

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X,y)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = log_reg.predict_proba(X_new)

plt.plot(X_new, y_prob[:,1],"g-",label="Iris virginica")
plt.plot(X_new, y_prob[:,0],"b--",label="Not Iris virginica")
plt.show()

X = iris["data"][:,(2,3)]
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10)
softmax_reg.fit(X,y)

sm_p1 = softmax_reg.predict([[5,2]])
sm_p2 = softmax_reg.predict_proba([[5,2]])

print(sm_p1)
print(sm_p2)




