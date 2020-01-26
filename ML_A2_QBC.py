import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import math

#stream-based
ini_frac = 0.22
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
#classes for entropy 
cl = np.unique(y)
entropy = [0.0]*X_val.shape[0]
#qbc
votes = [0]*cl.size
#models
sv = SVC(kernel='poly',degree=3,probability=True,gamma='auto')
sv.fit(X_train,y_train)
rf = RandomForestClassifier(n_estimators=50,max_depth=2)
rf.fit(X_train,y_train)
lr = LogisticRegression(solver='lbfgs',multi_class='multinomial')
lr.fit(X_train,y_train)
#initial accuracies
y_pred_sv = sv.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
accisv = sk.metrics.accuracy_score(y_pred_sv,y_test)
accirf = sk.metrics.accuracy_score(y_pred_rf,y_test)
accilr = sk.metrics.accuracy_score(y_pred_lr,y_test)
#stream
for x in X_val:
    sv_pred = sv.predict([x])
    rf_pred = rf.predict([x])
    lr_pred = rf.predict([x])
    print(sv_pred,lr_pred,rf_pred)

    for i in range(cl.size):
        if(sv_pred[0] == cl[i]):
            votes[i] += 1
        if(rf_pred[0] == cl[i]):
            votes[i] += 1
        if(lr_pred[0] == cl[i]):
            votes[i] += 1
    for i in range(cl.size):
        try:
            entropy[index] += votes[i]*(1/3)*(-1)*math.log(votes[i]*(1/3),2)
        except:
            entropy[index] = 0.0

    if(entropy[index]>0.5):
        X_train = np.append(X_train,[x],axis=0)
        y_train = np.append(y_train,[y_val[index]],axis = 0)
        
    index += 1
    #reset votes
    votes = [0]*cl.size

#re-fit classifiers based on vote entropy
#too time consuming to update model with each addition  
sv.fit(X_train,y_train)
rf.fit(X_train,y_train)
lr.fit(X_train,y_train)
y_pred_sv = sv.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
accsv = sk.metrics.accuracy_score(y_pred_sv,y_test)
accrf = sk.metrics.accuracy_score(y_pred_rf,y_test)
acclr = sk.metrics.accuracy_score(y_pred_lr,y_test)
print(accisv,accirf,accilr,accsv,accrf,acclr)