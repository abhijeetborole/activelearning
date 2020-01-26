import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#pool-based - Using SVM, Can use any classifier that generates probabilities
pool_frac = 0.03
max_iter = 4
label_frac = 0.2
#data
df = pd.read_csv('haberman.data',header=None)
df = df.sample(frac=1).reset_index(drop=True)
r = df.shape[0]
c = df.shape[1]
pool_size = int(r*pool_frac)
s = [0]*pool_size
Xf = df.iloc[:,:-1]
yf = df.iloc[:,-1:]
X = Xf.values
y = yf.values
#train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#random sample of labelled/unlabelled data from training dataset
chosen = np.random.choice(X_train_full.shape[0],int(X_train_full.shape[0]*label_frac))
X_train = X_train_full[chosen]
y_train = y_train_full[chosen]
X_val = np.delete(X_train_full,chosen,axis=0)
y_val = np.delete(y_train_full,chosen,axis=0)
#check initial accuracy of the model
sv = SVC(kernel='poly',degree=3,probability=True,gamma='auto')
sv.fit(X_train,y_train)
y_pred = sv.predict(X_test)
acci = sk.metrics.accuracy_score(y_pred,y_test)
     
for iter in range(max_iter):
    print('Iteration :'+str(iter+1))
    #probabilities of each class
    probas_val_sv = sv.predict_proba(X_val)
    #Least Confidence for Uncertainty Sampling
    prob_max = np.zeros(np.size(probas_val_sv,0))
    for i in range(np.size(probas_val_sv,0)):
        prob_max[i] = np.amax(probas_val_sv[i])
    for i in range(pool_size):
        s[i] = np.argmin(prob_max)
        prob_max[s[i]] = 1
    #updating training and validation datasets
    X_train = np.append(X_train, X_val[s], axis = 0)
    X_val = np.delete(X_val,s,axis=0)
    y_train = np.append(y_train, y_val[s], axis = 0)
    y_val = np.delete(y_val,s,axis=0)
    #train with updated dataset
    sv.fit(X_train,y_train)
    #predict
    y_pred = sv.predict(X_test)
    acc = sk.metrics.accuracy_score(y_pred,y_test)

print('Accuracy Increased from :'+str(acci)+' to :'+str(acc))








