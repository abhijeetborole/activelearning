import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#stream-based
ini_frac = 0.1
index = 0
#data
df = pd.read_csv('haberman.data',header=None)
df = df.sample(frac=1).reset_index(drop=True)
r = df.shape[0]
c = df.shape[1]
Xf = df.iloc[:,:-1]
yf = df.iloc[:,-1:]
X = Xf.values
y = yf.values
#train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#random sample of labelled/unlabelled data from training dataset
chosen = np.random.choice(X_train_full.shape[0],int(X_train_full.shape[0]*ini_frac))
X_train = X_train_full[chosen]
y_train = y_train_full[chosen]
X_val = np.delete(X_train_full,chosen,axis=0)
y_val = np.delete(y_train_full,chosen,axis=0)
#check initial accuracy of the model
sv = SVC(kernel='poly',degree=3,probability=True,gamma='auto')
sv.fit(X_train,y_train)
y_pred = sv.predict(X_test)
acci = sk.metrics.accuracy_score(y_pred,y_test)
for x in X_val:
    probas_val = sv.predict_proba([x])
    print(np.amax(probas_val))
    #threshold probability of least confidence for selection of model 
    if np.amax(probas_val) < 0.655:
        #update datasets and train
        X_train = np.append(X_train,[x],axis=0)
        y_train = np.append(y_train,[y_val[index]],axis = 0)
        sv.fit(X_train,y_train)
        print(index)

    index+=1
    y_pred = sv.predict(X_test)
    acc = sk.metrics.accuracy_score(y_pred,y_test)

print('Accuracy Increased from :'+str(acci)+' to :'+str(acc))








