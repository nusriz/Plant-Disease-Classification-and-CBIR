#############################################################################
Path to be changed to drive link location of features after added


###############################################################################
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import numpy  as np
import cv2
from sklearn.utils import shuffle
import os
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
########################  Constants   ############################
HEALTHY = -1
BACTERIAL_SPOT = 0
SEPTORIAL_SPOT = 1


########################  Variables   ############################
X = []
Y = []
y = []

#######################  Bacterial Spot ##########################
path = '/home/abishek/FYP/Final-Year-Project/Dataset/features/bacterialspot'

for file in os.listdir(path):
	img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
	arr = np.array(img)
	feature_vector = arr.flatten()
	X.append(feature_vector)
	y.append(BACTERIAL_SPOT)


#######################  Healthy leaf ############################
path = '/home/abishek/FYP/Final-Year-Project/Dataset/features/healthy'

for file in os.listdir(path):
	img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
	arr = np.array(img)
	feature_vector = arr.flatten()
	X.append(feature_vector)
	y.append(HEALTHY)




#######################  SEPTORIAL SPOTS ############################
path = '/home/abishek/FYP/Final-Year-Project/Dataset/features/septorial spots'

for file in os.listdir(path):
	img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
	arr = np.array(img)
	feature_vector = arr.flatten()
	X.append(feature_vector)
	y.append(SEPTORIAL_SPOT)



######################## Shuffle feature vector ###################
features = np.asarray(X)
Y=[y]
label    = np.asarray(Y)
print(features.shape)
print(label.shape)

concate = np.concatenate((features,label.T),axis=1)
rand = shuffle(concate,random_state=3)
#features,label, *rest = np.hsplit(rand,np.array([426400,2]))
label = rand[: ,426400]
#print(np.asarray(rand).shape)
#print(np.asarray(label).shape)

#label = rand[:,426400:]
#label = label.T
print(label)
features = rand[:, :426400]
print(label.shape)





X_train, X_test, y_train, y_test = train_test_split( features, label,  test_size = 0.25, random_state=32,stratify = label) 


pca = PCA(n_components = 2)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.fit_transform(X_test)


lda = LDA(n_components = 2)
lda_X_train2 = lda.fit_transform(X_train,y_train)
lda_X_test2 = lda.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train2, y_train) 
  
# Predict on dataset which model has not seen before 
print(knn.predict(X_test2)) 
print(y_test.T)

print(knn.score(X_test2, y_test))


#plot_decision_regions(X_train2,y_train, clf=knn, legend=2)# Adding axes annotations

##############################################################################
knn.fit(lda_X_train2, y_train) 
  
# Predict on dataset which model has not seen before 
print(knn.predict(lda_X_test2)) 
print(y_test.T)

print(knn.score(lda_X_test2, y_test))


plot_decision_regions(lda_X_train2,y_train, clf=knn, legend=2)# Adding axes annotations

plt.show()



