import pandas as pd
import numpy as np
import sklearn
import os
import time
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model, load_model
from dbn.models import SupervisedDBNClassification
from dbn.models import UnsupervisedDBN
from sklearn.svm import SVC, LinearSVC, NuSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from imports import *
from util import *

# data = PreProcess(DATASET_PATH)

# benign_df = pd.read_csv('./csv/benign.csv')
# malign_df = pd.read_csv('./csv/malign.csv')
df = pd.read_csv('./csv/static_dynamic_final.csv')

# concatenating dataframe of benign and malign
# frames = [benign_df, malign_df]
# df = pd.concat(frames)
# df.reset_index()
# Now df has the dataframe of features of all the apks 

# dropping the first 2 columns as the first 2 columns are useless
y = df['binary_label']
df.drop(['hash', 'label', 'binary_label'], axis=1, inplace=True)
X = df.values
length = X.shape[0]

print("Actual Dimenions of feature space is:")
print(X.shape)

# splitting the data into training and testing set
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8)

# Creating the auto-encoder 
# if LOAD_MODEL:
#     print("loading the encoder model...")
#     loaded_encoder = load_model("pickled/autoEncoder.mod")
#     loaded_encoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
# else:
#     encoding_dim = 1000  # dimension of the output
#     # this is our input placeholder
#     input_img = Input(shape=(X.shape[1],))
#     # "encoded" is the encoded representation of the input
#     encoded = Dense(encoding_dim, activation='relu')(input_img)
#     # "decoded" is the lossy reconstruction of the input
#     decoded = Dense(X.shape[1], activation='relu')(encoded)
#     # this model maps an input to its reconstruction
#     autoencoder = Model(input_img, decoded)
#
#     autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
#
#     autoencoder.fit(
#         X_train, X_train,
#         epochs=5,
#         batch_size=5,
#         shuffle=True,
#         validation_data=(X_test, X_test)
#     )
#
#     # saving the encoder with encoded as the output - we need this to readuce the dimension of the feature vector space
#     our_encoder = Model(input_img, encoded)
#     our_encoder.save('pickled/autoEncoder.mod')
#     loaded_encoder = our_encoder
#
# # reducing the dimension of the feature vector space from (12*186) to (12*8)
# X_with_reduced_dimension = loaded_encoder.predict(X)
# print('Now the dimensions of the encoded input feature space is -- ')
# print(X_with_reduced_dimension.shape)

if not LOAD_MODEL:
    time_start = time.time()
    # Creating the classifier for the reduced dimension input to classify 
    # the Clasifier is a hybrid model which comprises of Unsupervised DBN followed by a SVM Classfier to predict the label
    svm = SVC()
    dbn = SupervisedDBNClassification(hidden_layers_structure=[170, 140, 140, 2],
                          batch_size=100,
                          learning_rate_rbm=0.05,
                          n_epochs_rbm=20,
                          n_iter_backprop=30,
                          activation_function='relu')
    classifier = Pipeline(steps=[('dbn', dbn),
                                 ('svm', svm)])

    classifier.fit(X, y)

    time_end = time.time()
    print("Training cost time:")
    print(time_end-time_start)

    f = open("pickled/DBNClassifier_static_dynamic.pkl", "wb")
    pickle.dump(classifier, f)
    f.close()
f = open("pickled/DBNClassifier_static_dynamic.pkl", "rb")
classifier = pickle.load(f)
f.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
# df = pd.read_csv('./csv/static_features_500_test_20per.csv')
# y_test = df['binary_label']
# df.drop(['hash', 'label', 'binary_label'], axis=1, inplace=True)
# X_test = df.values
print("Classification_report is as follow:")
# print(loaded_encoder.predict(X_test))
# print(Y_test)
print("The accuracy_score is:")
print(accuracy_score(y_test, classifier.predict(X_test)))
print("--------------------------------------------------------------------")
print(classification_report(y_test, classifier.predict(X_test)))
print("classification_report end")

# def classify(apk):
#     print("Analyzing APK -- " + os.path.basename(apk))
#     print("-------------------------------------------------------------")
#     a, d, dx = AnalyzeAPK(apk)
#     feats = list()
#
#     feats += (data.makeHotVector([data.vocabPerm.index(p) for p in a.get_permissions() if p in data.vocabPerm], \
#                                  data.vocabLengths["perm"])).tolist()
#     feats += (data.makeHotVector([data.vocabServ.index(p) for p in a.get_services() if p in data.vocabServ], \
#                                  data.vocabLengths["serv"])).tolist()
#     feats += (data.makeHotVector([data.vocabRecv.index(p) for p in a.get_receivers() if p in data.vocabRecv], \
#                                  data.vocabLengths["recv"])).tolist()
#
#     test_feats = np.array(feats)
#     test_feats = test_feats.reshape((-1, 1))
#     print("features of the test apk after is ")
#     print(X_with_reduced_dimension.shape)
#     # print('Shape of the test apk features = ',test_feats.shape)
#
#     loaded_encoder = load_model("pickled/autoEncoder.mod")
#     loaded_encoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
#
#     test_feats_with_reduced_dimension = loaded_encoder.predict(test_feats.T)
#     print('Shape of the test apk features with readuced dimension = ')
#     print(test_feats_with_reduced_dimension.shape)
#
#     f = open("pickled/DBNClassifier.pkl", "rb")
#     classifier = pickle.load(f)
#     f.close()
#
#     print('The hybrid model prediction =', classifier.predict(test_feats_with_reduced_dimension))
#
#     if classifier.predict(test_feats_with_reduced_dimension) == [0]:
#         print("Malign")
#     else:
#         print("Bengin")
#
#
# print("-----------------------------------------------------------------------------------------------------")
# print("Classifying APKs from " + TEST_DATASET_PATH + " folder.")
# print("-----------------------------------------------------------------------------------------------------")
# print("Given input is malware, predicting ")
# classify(TEST_DATASET_PATH + "mal.apk")
# print("-------------------------------------------------------------")
# print("Given input is benign, predicting")
# classify(TEST_DATASET_PATH + "ben.apk")
