#classification of digits
import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784',version=1,cache=True,as_frame=False)
print(mnist.keys())

X,y = mnist["data"], mnist["target"]

print(X.shape,y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")
plt.show()

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000],X[60000:],y[:60000],y[60000:]

#Classify the '5' digits

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#SGD stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)

digit1p = sgd_clf.predict([some_digit])

print(digit1p)

from sklearn.model_selection import cross_val_score, cross_val_predict

cvs = cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy")
print(cvs)

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5,y_train_pred)

print(cm)

#Error Analysis

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
conf_mx = confusion_matrix(y_train,y_train_pred)
print(conf_mx)

plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx = conf_mx/row_sums

np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()






