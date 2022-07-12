# Imports
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Plot the data
#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,600,1)])
np.random.seed(10) #Setting seed for reproducability
y1 = np.sin(x) + np.random.normal(0,0.15,len(x))
y2 = np.cos(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y1,y2]),columns=['x','y1','y2'])


# Combine the data
y12 = np.append(y1, y2)
x12 = np.append(x, x)
y = np.append(np.repeat(1, len(x)), np.repeat(2, len(x)))
X = np.column_stack([x12, y12])

combined_data = pd.DataFrame(np.column_stack([X, y]),columns=['y','x','c'])
print(x12[1])



# Plot the results of SVM
def plot_result(X, model, kernel):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max - x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z)



# Question 2: SVM with radial kernel
from sklearn import svm
# You will use SVM with the following parameter kernel='rbf', C=1e3, gamma=0.01
model_radial = svm.SVC(kernel='rbf', C=1e3, gamma=0.01)
# fit the model
svc_radial = model_radial.fit(X,y)


# Question 3: SVM with sigmoid kernel
# You will use SVM with the following parameter kernel='sigmoid', C=1, gamma=0.01
model_sigmoid = svm.SVC(kernel='sigmoid', C=1, gamma=0.01)
# fit the model
svc_sigmoid = model_sigmoid.fit(X,y)


# Question 4: SVM with linear kernel
# You will use SVM with the following parameter  (kernel='linear', C=1, gamma=0.01)
model_lk = svm.SVC(kernel='linear', C=1, gamma=0.01)
# fit the model
svc_lk = model_lk.fit(X,y)


# Question 5: SVM with polynomial kernel
# You will use SVM with the following parameter kernel='poly', degree = 2, C=1, gamma=1e-3
model_pk = svm.SVC(kernel='poly', degree = 2, C=1, gamma=1e-3)
# fit the model
svc_pk = model_pk.fit(X,y)



# Question 6: SVM with polynomial kernel of different degree
# You will use SVM with the following parameter kernel='poly', degree = 3, C=1e3, gamma=1e-3
model_pkd = svm.SVC(kernel='poly', degree = 3, C=1e3, gamma=1e-3)
# fit the model
svc_pkd = model_pkd.fit(X,y)



def question2():
    print(model_radial.score(X, y))
    plot_result(X, svc_radial, "radial")

def question3():
    print(model_sigmoid.score(X, y))
    plot_result(X, svc_sigmoid, "sigmoid")

def question4():
    print(model_lk.score(X, y))
    plot_result(X, svc_lk, "linear")

def question5():
    print(model_pk.score(X, y))
    plot_result(X, svc_pk, "polynomial")


def question6():
    print(model_pkd.score(X, y))
    plot_result(X, svc_pkd, "polynomial")


if __name__ == '__main__':
    #func_name = "question1"
    func_name = input().strip()
    globals()[func_name]()
