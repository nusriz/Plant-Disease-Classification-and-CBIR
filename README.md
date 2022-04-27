# Plant-Disease-Classification-and-CBIR

This project is an attempt to develop a Content Based Image Retrieval (CBIR)
system to retrieve images of diseased leaves of tomato plant. It uses colour, shape
and texture features of the tomato leaf to classify and retrieve similar images. HSV
colour histogram is used to extract colour features. Fourier descriptors provides
shape feature, in the form of contour of the region of interest. Local Binary Pattern
(LBP) is widely used for texture extraction. In order to consider global texture
features, a variant of LBP called Completed Local Binary Pattern (CLBP) is utilized.
Here we find sign component and magnitude component using the differences
between the neighbouring pixels and its centre pixel. Also Uniform rotational invariant
LBP is applied to reduce the number of patterns. Furthermore feature fusion
of all colour, shape and texture properties is done to increase accuracy. Based
on this feature vector, classification of disease is done using a supervised learning
technique called Support Vector Machine (SVM). Analysis of different kernels like
linear, RBF, polynomial etc. and hyperparameter optimization showed that Linear
kernel is best suitable. Different combinations of features and their corresponding
accuracy is found to choose the best accurate model. Similar analysis is carried out
to find the suitable distance metric for retrieval purposes. In regards of classification,
mean accuracy of 97.3% is achieved in linear SVC model by 5-fold cross validation.
And in retrieval, mean average precision of 85.08% and Bull’s eye performance of
90.17% are obtained using canberra distance metric.
Keywords : Colour Histograms, Fourier Descriptors, Local Binary Patterns, Completed
LBP, Support Vector Machine, Content Based Image Retrieval
