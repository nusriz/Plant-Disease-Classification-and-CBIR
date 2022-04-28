import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
import os
from numpy import savetxt,loadtxt

################# Files  ##########################
# 1: bgr hist
# 2: hsv hist
# 3: shape
# 4: gray_clbp_S_M
# 5: hue_clbp_S_M

################# Constants #######################
B_SPOTS = -2
HEALTHY = -1
L_BLIGHT = 0
T_MOSAIC = 1
S_SPOTS = 2 
diseases = ['bspot','healthy','lblight','mosaic','sspot']
label_dict = {'bspot':B_SPOTS,'healthy':HEALTHY,'lblight':L_BLIGHT,'mosaic':T_MOSAIC,'sspot':S_SPOTS}
files = [1,2,3,4,5]

################ Variables #########################
X = []
Y = []
y = []

################ Feature Vector ####################
path = 'drive/My Drive/Features'
flag = 0
flag2 = 0
for disease in diseases:
  read_path = os.path.join(path,disease)
  for file in files:
    f_name = str(file)+'.csv'
    file_path = os.path.join(read_path,f_name)

    if flag == 0:
      data = loadtxt(file_path,delimiter = ',')
      data = np.asarray(data)
    
    if len(files) >=2 and flag == 1:
      data1 = loadtxt(file_path,delimiter = ',')
      data1 = np.asarray(data1)
      data = np.concatenate((data,data1),axis = 1)

    
    
    flag = 1
  print(data.shape)
  flag = 0 
  temp = np.zeros(60)
  temp.fill(label_dict[disease])
  if flag2 == 0:
    feature_data = data
    label = temp
  if flag2 == 1:
    feature_data = np.concatenate((feature_data,data))
    label = np.concatenate((label,temp))
  flag2 = 1  

features = feature_data
print(feature_data.shape)
print(label.shape)  

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
#clf = svm.SVC(gamma = 0.001, C = 0.01)
X_train, X_test, y_train, y_test = train_test_split( features, label,  test_size = 0.3, random_state=10,stratify = label) 
#clf.fit(X_train,y_train)
#print(clf.predict(X_test)) 
#print(y_test.T)
#
#print(clf.score(X_test, y_test))
#
knn = KNeighborsClassifier(n_neighbors=7) 
knn.fit(X_train, y_train) 
  
# # Predict on dataset which model has not seen before 
#print(knn.predict(X_test)) 
#print(y_test.T)

print(knn.score(X_test, y_test))

m = OneVsRestClassifier(LinearSVC(max_iter=1000,dual = False))
m.fit(X_train,y_train)
#print(m.predict(X_test))
#print(y_test.T)

print(m.score(X_test, y_test))

j = LinearSVC(max_iter=1000,dual = False)
j.fit(X_train,y_train)
print(j.score(X_test,y_test))


from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
m = OneVsRestClassifier(LinearSVC(max_iter=10000))
clf = make_pipeline(StandardScaler(), logisticRegr)
warnings.filterwarnings('ignore')

cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.7 , random_state = 4)
print(cv)
score = cross_val_score(clf,features,label,cv=cv)
print(score)
print(score.mean())

from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
m = OneVsRestClassifier(LinearSVC(max_iter=10000))
clf = make_pipeline(StandardScaler(), logisticRegr)
warnings.filterwarnings('ignore')

cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.7 , random_state = 4)
print(cv)
score = cross_val_score(clf,features,label,cv=cv)
print(score)
print(score.mean())

from sklearn.preprocessing import StandardScaler
fit_data = features
scaler = StandardScaler()
print(scaler.fit(fit_data))
t_feat = scaler.transform(fit_data)
X_train, X_test, y_train, y_test = train_test_split( t_feat, label,  test_size = 0.4, random_state=36,stratify = label)
m = OneVsRestClassifier(LinearSVC())
m.fit(X_train,y_train)
print(m.predict(X_test))
print(y_test.T)

print(m.score(X_test, y_test))

from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pca = PCA(n_components = 2)
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.fit_transform(X_test)
lda = LDA(n_components = 2)
lda_X_train = lda.fit_transform(X_train,y_train)
lda_X_test = lda.transform(X_test)

m = OneVsRestClassifier(LinearSVC())
m.fit(pca_X_train,y_train)
print(m.predict(pca_X_test))
print(y_test.T)

print(m.score(pca_X_test, y_test))

m = OneVsRestClassifier(LinearSVC())
m.fit(lda_X_train,y_train)
print(m.predict(lda_X_test))
print(y_test.T)

print(m.score(lda_X_test, y_test))

