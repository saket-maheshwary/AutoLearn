#!/usr/bin/env python

import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVR
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge
import matplotlib.patches as mpatches
from sklearn.linear_model import perceptron
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RandomizedLasso
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif

__author__ = "Saket Maheshwary, Ambika Kaul"
__credits__ = ["Saket Maheshwary", "Ambika Kaul"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Saket Maheshwary, Ambika Kaul"
__email__ = "saketm.1991@gmail.com"
__status__ = "Code Optimization is in Progress"

warnings.filterwarnings('ignore')
#clf=linear_model.LinearRegression()
clf = Ridge(alpha=1.0)
cnt=0
ans=[]

np.random.seed(7)  # to ensure that everytime results are same

###########################################################################
                  # Function to randomly shuffle the data
###########################################################################


def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
      df.apply(np.random.shuffle, axis=axis)
    return df


###########################################################################
                   # Function to define quadratic curve
###########################################################################


def curve(x,a,b,c):
    return a*(x**2) + b*x + c


###########################################################################
              # Function to compute the distance correlation
###########################################################################


def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


###########################################################################
       # Compute the linearly and non-linearly correlated features
###########################################################################


def dependent(x,th1,fold):
    ans=[]
    ans1=[]
    m,n=x.shape
    cnt=0
    cnt1=0
    for i in range(0,n):
       for j in range(0,n):
           if (i!=j):
              a,b=pearsonr(x[:,i][:,np.newaxis],x[:,j][:,np.newaxis])
              if(distcorr(np.array(x[:,i]),np.array(x[:,j]))>=th1):
               a1=i,j
               ans.append(a1)
               cnt=cnt+1
              elif(distcorr(np.array(x[:,i]),np.array(x[:,j]))>0 and distcorr(np.array(x[:,i]),np.array(x[:,j]))<0.7):
               zz=i,j
               ans1.append(zz)
               cnt1=cnt1+1

       #print(i)
    if os.path.exists('sonar_linear_correlated_{}.csv'.format(fold)):                             # Name of Ouput file generated
       os.remove('sonar_linear_correlated_{}.csv'.format(fold))
    if os.path.exists('sonar_nonlinear_correlated_{}.csv'.format(fold)):                          # Name of Ouput file generated
       os.remove('sonar_nonlinear_correlated_{}.csv'.format(fold))

    np.savetxt("sonar_linear_correlated_{}.csv".format(fold),ans,delimiter=",",fmt="%.5f")
    np.savetxt("sonar_nonlinear_correlated_{}.csv".format(fold),ans1,delimiter=",",fmt="%s")

    print("This is fold no - {}".format(fold))
    print("Number of linear correlated features are:")
    print(cnt)
    print("Number of non linear correlated features are:")
    print(cnt1)


###########################################################################
     # Function to compute the relative feature importance of features
###########################################################################


def rank(X1,y):
    forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
    forest.fit(X1, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

# Print the feature ranking
    print("Feature ranking:")

    for f in range(X1.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
    plt.figure()
    listt=plt.bar(range(X1.shape[1]), importances[indices], align="center")
    for xx in range(X1.shape[1]):
        if(xx>0 and xx<60):
           listt[indices[xx]].set_color('r')
    #plt.xticks(range(X1.shape[1]),'')
    plt.xlim([-1, X1.shape[1]])
    plt.ylabel('Relative Importance')
    plt.xlabel('Number of Features')
    red_patch = mpatches.Patch(color='red', label='Original Features')
    blue_patch = mpatches.Patch(color='blue', label='Newly Constructed Features')
    plt.legend(handles=[red_patch,blue_patch],loc='upper right')
    #plt.legend(('Constructed Features importance'), shadow=True)
    #plt.legend('Actual Features importance', shadow=True)
    #plt.legend(['Original Features'], loc='upper right')
    #plt.legend(['New Features'], loc='upper right')
    #plt.show()


###########################################################################
          # Predicting feature values using linear Regression
###########################################################################


def linear(TR,TST,fold):
    a,b=TR.shape
    c,d=TST.shape
    dataset=pd.read_csv('sonar_linear_correlated_{}.csv'.format(fold),header=None)
    val=dataset.as_matrix(columns=None)
    aa,bb=val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train=np.zeros((a,len(ans)),dtype=float)
    predicted_test=np.zeros((c,len(ans)),dtype=float)
    predicted_train_final=np.zeros(2*(a,len(ans)),dtype=float)
    predicted_test_final=np.zeros(2*(c,len(ans)),dtype=float)
    predicted_train_error=np.zeros((a,len(ans)),dtype=float)
    predicted_test_error=np.zeros((c,len(ans)),dtype=float)

    for j in range(0,aa):
           rr,ss=np.array(TR[:,(int)(val[j][0])][:,np.newaxis]),np.array(TR[:,(int)(val[j][1])])
           tt,uu=np.array(TST[:,(int)(val[j][0])][:,np.newaxis]),np.array(TST[:,(int)(val[j][1])])
           y_train=clf.fit(rr,ss).predict(rr)[:,np.newaxis]
           y_test=clf.fit(rr,ss).predict(tt)[:,np.newaxis]
           predicted_train=np.hstack([predicted_train,y_train])
           predicted_test=np.hstack([predicted_test,y_test])

           dd=ss[:,np.newaxis]
           ee=uu[:,np.newaxis]
           diff_train=(dd-y_train)
           diff_test=(ee-y_test)
           predicted_train_error=np.hstack([predicted_train_error,diff_train])
           predicted_test_error=np.hstack([predicted_test_error,diff_test])
           #predicted_test=np.hstack([predicted_test,clf.coef_*TST[:,val[j][1]][:, np.newaxis]+clf.intercept_])
           #predicted_train=np.hstack([predicted_train,clf.coef_*TR[:,val[j][1]][:, np.newaxis]+clf.intercept_])
    predicted_train_final=np.hstack([predicted_train,predicted_train_error])
    predicted_test_final=np.hstack([predicted_test,predicted_test_error])
    # Saving constructed features finally to a file

    if os.path.exists("sonar_related_lineartest_{}.csv".format(fold)):                          # Name of Ouput file generated
       os.remove("sonar_related_lineartest_{}.csv".format(fold))

    if os.path.exists('sonar_related_lineartrain_{}.csv'.format(fold)):                          # Name of Ouput file generated
       os.remove('sonar_related_lineartrain_{}.csv'.format(fold))

    with open("sonar_related_lineartest_{}.csv".format(fold), "wb") as myfile:
            np.savetxt(myfile,predicted_test_final,delimiter=",",fmt="%s")
    with open("sonar_related_lineartrain_{}.csv".format(fold), "wb") as myfile:
            np.savetxt(myfile,predicted_train_final,delimiter=",",fmt="%s")


###########################################################################
        # Predicting feature values using Non linear Regression
###########################################################################


def nonlinear(TR,TST,fold):
    a,b=TR.shape
    c,d=TST.shape
    dataset=pd.read_csv('sonar_nonlinear_correlated_{}.csv'.format(fold),header=None)
    val=dataset.as_matrix(columns=None)
    aa,bb=val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train=np.zeros((a,len(ans)),dtype=float)
    predicted_test=np.zeros((c,len(ans)),dtype=float)
    predicted_train_final=np.zeros(2*(a,len(ans)),dtype=float)
    predicted_test_final=np.zeros(2*(c,len(ans)),dtype=float)

    predicted_train_error=np.zeros((a,len(ans)),dtype=float)
    predicted_test_error=np.zeros((c,len(ans)),dtype=float)

    svr_rbf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',
                          kernel_params=None)

    #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    for j in range(0,aa):
        rr,ss=np.array(TR[:,(int)(val[j][0])][:,np.newaxis]),np.array(TR[:,(int)(val[j][1])])
        tt,uu=np.array(TST[:,(int)(val[j][0])][:,np.newaxis]),np.array(TST[:,(int)(val[j][1])])

        y_train=svr_rbf.fit(rr,ss).predict(rr)[:,np.newaxis]
        y_test=svr_rbf.fit(rr,ss).predict(tt)[:,np.newaxis]
        predicted_train=np.hstack([predicted_train,y_train])
        predicted_test=np.hstack([predicted_test,y_test])


        dd=ss[:,np.newaxis]
        ee=uu[:,np.newaxis]
        diff_train=(dd-y_train)
        diff_test=(ee-y_test)

        predicted_train_error=np.hstack([predicted_train_error,diff_train])
        predicted_test_error=np.hstack([predicted_test_error,diff_test])
        '''
        popt, pcov = curve_fit(curve,rr,ss)
        predicted_test=np.hstack([predicted_test,float(popt[0])*(TST[:,val[j][1]][:, np.newaxis]**2)+float(popt[1])*TST[:,val[j][1]][:,    np.newaxis]+float(popt[2])])
        predicted_train=np.hstack([predicted_train,float(popt[0])*(TR[:,val[j][1]][:, np.newaxis]**2)+float(popt[1])*TR[:,val[j][1]][:, np.newaxis]+float(popt[2])])
        '''
    predicted_train_final=np.hstack([predicted_train,predicted_train_error])
    predicted_test_final=np.hstack([predicted_test,predicted_test_error])

    if os.path.exists("sonar_related_nonlineartest_{}.csv".format(fold)):                          # Name of Ouput file generated
       os.remove("sonar_related_nonlineartest_{}.csv".format(fold))

    if os.path.exists('sonar_related_nonlineartrain_{}.csv'.format(fold)):                          # Name of Ouput file generated
       os.remove('sonar_related_nonlineartrain_{}.csv'.format(fold))

    # Saving constructed features finally to a file
    with open("sonar_related_nonlineartest_{}.csv".format(fold), "wb") as myfile:
            np.savetxt(myfile,predicted_test_final,delimiter=",")
    with open("sonar_related_nonlineartrain_{}.csv".format(fold), "wb") as myfile:
            np.savetxt(myfile,predicted_train_final,delimiter=",")


#############################################################################
    # Function to select features from the newly constructed ones
#############################################################################


def stable(ress,test,labels):   # ress is training data
    x,y = ress.shape
    names = np.arange(y)
    rlasso = RandomizedLasso()
    rlasso.fit(ress,labels)

    #print "Features sorted by their scores according to the stability scoring function"
    val=sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
    			 names), reverse=True)

    print("len of val")  # newly constructed features
    print(len(val))
    global nc_val
    nc_val += len(val)

    finale=[]
    for i in range(0,len(val)):
        r,s=val[i]   # 'r' represents scores, 's' represents column name
        if(r>0.1):   # This is eta for stability selection
          finale.append(s)

        #finale.append(s)

    print("Total features after stability selection:")
    print(len(finale))  # finale stores col names - 2nd, 4th etc of stable features.
    global stable_val
    stable_val += len(finale)

    dataset1=np.zeros((len(ress),len(finale)),dtype=float)
    dataset3=np.zeros((len(test),len(finale)),dtype=float)
    dataset1=ress[:,finale]
    dataset3=test[:,finale]
    #dataset3=test.iloc[:,finale]

    if os.path.exists("sonar_stable_testfeatures.csv"):                           # Name of Ouput file generated
       os.remove("sonar_stable_testfeatures.csv")
    if os.path.exists("sonar_stable_trainfeatures.csv"):                          # Name of Ouput file generated
       os.remove("sonar_stable_trainfeatures.csv")

    with open("sonar_stable_testfeatures.csv", "wb") as myfile:
            np.savetxt(myfile,dataset3,delimiter=",",fmt="%s")
    with open("sonar_stable_trainfeatures.csv", "wb") as myfile:
            np.savetxt(myfile,dataset1,delimiter=",",fmt="%s")

#-----------------------------------------------------------------------------------
    # check the inter-feature dependence - 2nd phase of ensemble

    ress_new = SelectKBest(mutual_info_classif, k='all')
    ress_new.fit_transform(ress[:,finale], labels)

    #print "Features sorted by their scores according to the scoring function - mutual information gain:"
    feats=sorted(zip(map(lambda x: round(x, 4), ress_new.scores_),
                 names), reverse=True)

    ensemble_finale=[]
    for i in range(0,len(feats)):
        r,s=feats[i]
        if(r>0):   # This is eta-o
          ensemble_finale.append(s)


    print("Total features after 2 phase selection:")
    print(len(ensemble_finale))  # ensemble_finale stores col names further pruned in the 2nd phase of feature selection
    global ensemble_val
    ensemble_val += len(ensemble_finale)
    #print(ensemble_select)

    dataset2=np.zeros((len(ress),len(ensemble_finale)),dtype=float)
    dataset4=np.zeros((len(test),len(ensemble_finale)),dtype=float)
    dataset2=ress[:,ensemble_finale]
    dataset4=test[:,ensemble_finale]

    if os.path.exists("sonar_ensemble_testfeatures.csv"):                           # Name of Ouput file generated
       os.remove("sonar_ensemble_testfeatures.csv")
    if os.path.exists("sonar_ensemble_trainfeatures.csv"):                          # Name of Ouput file generated
       os.remove("sonar_ensemble_trainfeatures.csv")

    with open("sonar_ensemble_testfeatures.csv", "wb") as myfile:
            np.savetxt(myfile,dataset4,delimiter=",",fmt="%s")
    with open("sonar_ensemble_trainfeatures.csv", "wb") as myfile:
            np.savetxt(myfile,dataset2,delimiter=",",fmt="%s")


#############################################################################
    # Function to select features from the original ones using I.G
#############################################################################


def original_ig(ress,test,labels):   # ress is training data
    x,y = ress.shape
    names = np.arange(y)

    ress_new = SelectKBest(mutual_info_classif, k='all')
    ress_new.fit_transform(ress, labels)

    #print "Features sorted by their scores according to the scoring function - mutual information gain:"
    original_features=sorted(zip(map(lambda x: round(x, 4), ress_new.scores_),
    			 names), reverse=True)

    finale=[]
    for i in range(0,len(original_features)):
        r,s=original_features[i]
        if(r>0):   # This is eta-o
          finale.append(s)

        #finale.append(s)

    print("Selected features after O + IG:")
    global len_orig_ig
    len_orig_ig += len(finale)
    print(len(finale))
    dataset1=np.zeros((len(ress),len(finale)),dtype=float)
    dataset3=np.zeros((len(test),len(finale)),dtype=float)
    dataset1=ress[:,finale]
    dataset3=test[:,finale]
    #dataset3=test.iloc[:,finale]

    if os.path.exists("sonar_original_ig_testfeatures.csv"):                           # Name of Ouput file generated
       os.remove("sonar_original_ig_testfeatures.csv")
    if os.path.exists("sonar_original_ig_trainfeatures.csv"):                          # Name of Ouput file generated
       os.remove("sonar_original_ig_trainfeatures.csv")

    with open("sonar_original_ig_testfeatures.csv", "wb") as myfile:
            np.savetxt(myfile,dataset3,delimiter=",",fmt="%s")
    with open("sonar_original_ig_trainfeatures.csv", "wb") as myfile:
            np.savetxt(myfile,dataset1,delimiter=",",fmt="%s")

###############################################################################################################################################
                                                           # Main Function
###############################################################################################################################################


if __name__ == "__main__":
   df=pd.read_csv('sonar.csv',header=None)    # Name of the input numeric feature file in .csv format
   shuffle(df)
   data=df.sample(frac=1)
   n,m=data.shape
   print(n,m)

   x=data.drop(data.columns[len(data.columns)-1],1)
   Y=data[data.columns[len(data.columns)-1]]

   X=x.as_matrix()
   y=Y.as_matrix()
   print("Features in Original Dataset:")
   p,pp=X.shape
   print(pp)

   len_orig_ig=0
   nc_val=0
   stable_val=0
   ensemble_val=0
   # Dividing data into 5 parts where 4 parts are used for training and 1 for testing in each iteration

   train1=X[:(int)(0.8*n),:]
   test1=X[(int)(0.8*n):,:]

   train2=X[(int)(0.2*n):,:]
   test2=X[:(int)(0.2*n),:]

   train3=np.concatenate((X[:(int)(0.6*n),:],X[(int)(0.8*n):,:]),axis=0)
   test3=X[(int)(0.6*n):(int)(0.8*n),:]

   train4=np.concatenate((X[:(int)(0.4*n),:],X[(int)(0.6*n):,:]),axis=0)
   test4=X[(int)(0.4*n):(int)(0.6*n),:]

   train5=np.concatenate((X[:(int)(0.2*n),:],X[(int)(0.4*n):,:]),axis=0)
   test5=X[(int)(0.2*n):(int)(0.4*n),:]

   train1Y=y[:(int)(0.8*n)]
   test1Y=y[(int)(0.8*n):]

   train2Y=y[(int)(0.2*n):]
   test2Y=y[:(int)(0.2*n)]

   list1=y[:(int)(0.6*n)]
   list2=y[(int)(0.8*n):]
   train3Y=np.append(list1,list2)
   test3Y=y[(int)(0.6*n):(int)(0.8*n)]

   list1=y[:(int)(0.4*n)]
   list2=y[(int)(0.6*n):]
   train4Y=np.append(list1,list2)
   test4Y=y[(int)(0.4*n):(int)(0.6*n)]

   list1=y[:(int)(0.2*n)]
   list2=y[(int)(0.4*n):]
   train5Y=np.append(list1,list2)
   test5Y=y[(int)(0.2*n):(int)(0.4*n)]

   original={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
   			 'AdaBoost':0,'Neural Network':0,'Decision Tree':0}
   orig_ig={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
   				'AdaBoost':0,'Neural Network':0,'Decision Tree':0}
   new={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,'AdaBoost':0,\
        'Neural Network':0,'Decision Tree':0}
   new_fs={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,'AdaBoost':0,\
        'Neural Network':0,'Decision Tree':0}
   supplement={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
   			   'AdaBoost':0,'Neural Network':0,'Decision Tree':0}
   supplement_ig={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
                  'AdaBoost':0,'Neural Network':0,'Decision Tree':0}
   stable_ig={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
                  'AdaBoost':0,'Neural Network':0,'Decision Tree':0}


   #############################################################################
                # Computing Accuracy for each fold of Cross Validation
   #############################################################################

   original_ig(train1,test1,train1Y)  # No normalization needed for original training & testing
   original_ig_train1=pd.read_csv('sonar_original_ig_trainfeatures.csv', header=None)
   original_ig_test1=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train1=original_ig_train1.as_matrix()
   original_ig_test1=original_ig_test1.as_matrix()

   dependent(original_ig_train1, 0.7, 1)
   linear(original_ig_train1, original_ig_test1, 1)
   nonlinear(original_ig_train1, original_ig_test1, 1)

   a1=pd.read_csv('sonar_related_lineartest_1.csv',header=None)          # all predicted feature files
   a2=pd.read_csv('sonar_related_lineartrain_1.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_1.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_1.csv',header=None)

   #r4=a4
   #r3=a3
   r4=np.hstack([a2,a4])      # Train
   r3=np.hstack([a1,a3])      # Test

   scaler=StandardScaler().fit(r4) # Normalization  & fit only on training
   p2=scaler.transform(r4)     # Normalized Train
   p1=scaler.transform(r3)     # Normalized Test

   stable(p2,p1,train1Y)
   f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test1, f2])  # original test features, selected by IG, f2 is feature space after ensemble selection.
   x2X=np.hstack([train1, f1])

   scaler=StandardScaler().fit(x2X)  # Again normalization of the complete combined feature pool
   x2=scaler.transform(x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test1, f2])
   y2Y=np.hstack([train1, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)

   st_x1X=np.hstack([original_ig_test1, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train1, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("............................................................................................................................")

   print("Predicting Accuracies")

   names=['kNN','Logistic Regression','Linear SVM','Poly SVM','Random Forest','AdaBoost','Neural Network','Decision Tree']
   models=[KNeighborsClassifier(), LogisticRegression(), svm.LinearSVC(),SVC(C=1.0, kernel='poly'),
           RandomForestClassifier(),AdaBoostClassifier(), MLPClassifier(), tree.DecisionTreeClassifier()]

   print("....................Results on Original Features...............................")

   for i in range(0,len(models)):
      models[i].fit(train1,train1Y)
      y_out= models[i].predict(test1)
      print(models[i].score(test1,test1Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test1,test1Y)

   print("....................Results on (Original + IG) Stable Features...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train1,train1Y)
      y_out= models[i].predict(original_ig_test1)
      print(models[i].score(original_ig_test1,test1Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test1, test1Y)

   print("...................Results on Newly constructed Features.........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train1Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test1Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test1Y)

   print("...................Results after R2R.........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train1Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test1Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test1Y)

   print("...................Results on (5).............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train1Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test1Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test1Y)

   print("...................Results on (6).............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train1Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test1Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test1Y)

   print("...................Results on (7).............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train1Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test1Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test1Y)

   rank(x2,train1Y) # - rank function is for plotting graph - sec 5 in paper

   print("################################################################################")
   print("################################################################################")

   original_ig(train2,test2,train2Y)  # No normalization needed for original training & testing
   original_ig_train2=pd.read_csv('sonar_original_ig_trainfeatures.csv',header=None)
   original_ig_test2=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train2=original_ig_train2.as_matrix()
   original_ig_test2=original_ig_test2.as_matrix()

   dependent(original_ig_train2, 0.7, 2)
   linear(original_ig_train2, original_ig_test2, 2)
   nonlinear(original_ig_train2, original_ig_test2, 2)

   a1=pd.read_csv('sonar_related_lineartest_2.csv',header=None)
   a2=pd.read_csv('sonar_related_lineartrain_2.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_2.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_2.csv',header=None)

   r4=np.hstack([a2,a4])
   r3=np.hstack([a1,a3])
   #r4=a4
   #r3=a3

   scaler=StandardScaler().fit(r4)
   p2=scaler.transform(r4)
   p1=scaler.transform(r3)

   stable(p2,p1,train2Y)
   f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test2,f2])
   x2X=np.hstack([train2,f1])

   scaler=StandardScaler().fit(x2X)
   x2=scaler.transform(x2X)
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test2, f2])
   y2Y=np.hstack([train2, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)

   st_x1X=np.hstack([original_ig_test2, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train2, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("Predicting Accuracies")
   print("....................Results on Original Features...............................")

   for i in range(0,len(models)):
      models[i].fit(train2,train2Y)
      y_out= models[i].predict(test2)
      print(models[i].score(test2,test2Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test2,test2Y)

   print("....................Results on (Original + IG) Stable Features...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train2,train2Y)
      y_out= models[i].predict(original_ig_test2)
      print(models[i].score(original_ig_test2,test2Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test2,test2Y)

   print("...................Results on Newly constructed Features.........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train2Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test2Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test2Y)

   print("...................Results after R2R.........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train2Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test2Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test2Y)

   print("...................Results on (5).............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train2Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test2Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test2Y)

   print("...................Results on (6).............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train2Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test2Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test2Y)

   rank(x2,train2Y)

   print("...................Results on (7).............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train2Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test2Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test2Y)

   print("################################################################################")
   print("################################################################################")

   original_ig(train5,test5,train5Y)  # No normalization needed for original training & testing
   original_ig_train5=pd.read_csv('sonar_original_ig_trainfeatures.csv',header=None)
   original_ig_test5=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train5=original_ig_train5.as_matrix()
   original_ig_test5=original_ig_test5.as_matrix()

   dependent(original_ig_train5, 0.7, 5)
   linear(original_ig_train5, original_ig_test5, 5)
   nonlinear(original_ig_train5, original_ig_test5, 5)

   a1=pd.read_csv('sonar_related_lineartest_5.csv',header=None)
   a2=pd.read_csv('sonar_related_lineartrain_5.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_5.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_5.csv',header=None)
   #r4=a4
   #r3=a3
   r4=np.hstack([a2,a4])
   r3=np.hstack([a1,a3])

   scaler=StandardScaler().fit(r4)
   p2=scaler.transform(r4)
   p1=scaler.transform(r3)

   stable(p2,p1,train5Y)
   f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test5,f2])
   x2X=np.hstack([train5,f1])

   scaler=StandardScaler().fit(x2X)
   x2=scaler.transform(x2X)
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test5, f2])
   y2Y=np.hstack([train5, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)

   st_x1X=np.hstack([original_ig_test5, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train5, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("Predicting Accuracies")
   print(".................... Results on Original Features ...............................")

   for i in range(0,len(models)):
      models[i].fit(train5,train5Y)
      y_out= models[i].predict(test5)
      print(models[i].score(test5,test5Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test5,test5Y)

   print("....................Results on (Original + IG )Stable Features...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train5,train5Y)
      y_out= models[i].predict(original_ig_test5)
      print(models[i].score(original_ig_test5,test5Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test5,test5Y)

   print("...................Results only on Newly constructed Features.........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train5Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test5Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test5Y)

   print("...................Results after R2R.........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train5Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test5Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test5Y)

   print("...................Results on (5).............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train5Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test5Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test5Y)

   print("...................Results on (6).............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train5Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test5Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test5Y)

   print("...................Results when full architecture of AutoLearn is followed.............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train5Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test5Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test5Y)

   rank(x2,train5Y)
   print("################################################################################")
   print("################################################################################")


   original_ig(train4,test4,train4Y)  # No normalization needed for original training & testing
   original_ig_train4=pd.read_csv('sonar_original_ig_trainfeatures.csv',header=None)
   original_ig_test4=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train4=original_ig_train4.as_matrix()
   original_ig_test4=original_ig_test4.as_matrix()

   dependent(original_ig_train4,0.7, 4)
   linear(original_ig_train4,original_ig_test4, 4)
   nonlinear(original_ig_train4,original_ig_test4, 4)

   a1=pd.read_csv('sonar_related_lineartest_4.csv',header=None)
   a2=pd.read_csv('sonar_related_lineartrain_4.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_4.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_4.csv',header=None)

   #r4=a4
   #r3=a3
   r4=np.hstack([a2,a4])
   r3=np.hstack([a1,a3])
   scaler=StandardScaler().fit(r4)
   p2=scaler.transform(r4)
   p1=scaler.transform(r3)

   stable(p2,p1,train4Y)
   f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test4,f2])
   x2X=np.hstack([train4,f1])

   scaler=StandardScaler().fit(x2X)
   x2=scaler.transform(x2X)
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test4, f2])
   y2Y=np.hstack([train4, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)

   st_x1X=np.hstack([original_ig_test4, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train4, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("Predicting Accuracies")
   print("....................Results on Original Features...............................")

   for i in range(0,len(models)):
      models[i].fit(train4,train4Y)
      y_out= models[i].predict(test4)
      print(models[i].score(test4,test4Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test4,test4Y)

   print("....................Results on (Original + IG) Stable Features...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train4,train4Y)
      y_out= models[i].predict(original_ig_test4)
      print(models[i].score(original_ig_test4,test4Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test4,test4Y)

   print("...................Results only on Newly constructed Features.........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train4Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test4Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test4Y)

   print("...................Results after R2R.........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train4Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test4Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test4Y)

   print("...................Results on (5).............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train4Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test4Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test4Y)

   print("...................Results on (6).............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train4Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test4Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test4Y)

   print("................... Results when full architecture of AutoLearn is followed .............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train4Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test4Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test4Y)

   rank(x2,train4Y)
   print("################################################################################")
   print("################################################################################")

   original_ig(train3,test3,train3Y)  # No normalization needed for original training & testing
   original_ig_train3=pd.read_csv('sonar_original_ig_trainfeatures.csv',header=None)
   original_ig_test3=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train3=original_ig_train3.as_matrix()
   original_ig_test3=original_ig_test3.as_matrix()

   dependent(original_ig_train3, 0.7, 3)
   linear(original_ig_train3,original_ig_test3, 3)
   nonlinear(original_ig_train3,original_ig_test3, 3)

   a1=pd.read_csv('sonar_related_lineartest_3.csv',header=None)
   a2=pd.read_csv('sonar_related_lineartrain_3.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_3.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_3.csv',header=None)
   #r4=a4
   #r3=a3
   r4=np.hstack([a2,a4])
   r3=np.hstack([a1,a3])
   scaler=StandardScaler().fit(r4)
   p2=scaler.transform(r4)
   p1=scaler.transform(r3)

   stable(p2,p1,train3Y)
   f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test3,f2])
   x2X=np.hstack([train3,f1])

   scaler=StandardScaler().fit(x2X)
   x2=scaler.transform(x2X)
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test3, f2])
   y2Y=np.hstack([train3, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)

   st_x1X=np.hstack([original_ig_test3, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train3, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("Predicting Accuracies")
   print(".................... Results on Original Features ...............................")

   for i in range(0,len(models)):
      models[i].fit(train3,train3Y)
      y_out= models[i].predict(test3)
      print(models[i].score(test3,test3Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test3,test3Y)

   print(".................... Results on (Original + IG) Stable Features ...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train3,train3Y)
      y_out= models[i].predict(original_ig_test3)
      print(models[i].score(original_ig_test3,test3Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test3,test3Y)

   print("................... Results on just the Newly constructed Features .........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train3Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test3Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test3Y)

   print("................... Results after Newly constructed Features with feature selection .........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train3Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test3Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test3Y)

   print("................... Results on (5) .............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train3Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test3Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test3Y)

   print("................... Results when full architecture of AutoLearn is followed (IG & then Stability in feature selection) .............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train3Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test3Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test3Y)

   rank(x2,train3Y)
   print("................... Results when full architecture of AutoLearn is followed (Stability only in feature selection) .............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train3Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test3Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test3Y)


   print("################################################################################")
   print("################################################################################")
   #rank(Train1,y_train)
   #rank(Train,y_train)
   '''
   print("Original features", pp)
   print("Selected after IG (Avg)", len_orig_ig/5)
   print("---------------------------------------------")
   print("New Features Constructed (Avg)", nc_val/5)
   print("Features Selected after Stability Selection(Avg)", stable_val/5)
   print("---------------------------------------------")
   print("Features selected after ensemble (Avg)", ensemble_val/5)
   '''
   
   print("Accuracies :")

   print("................... Average of results after 5 fold CV in the same order as above .............................")


   for i in range(0,len(models)):
       print(names[i])
       print((original[names[i]]/5)*100)
       print((orig_ig[names[i]]/5)*100)
       print((new[names[i]]/5)*100)
       print((new_fs[names[i]]/5)*100)
       print((supplement[names[i]]/5)*100)
       print((supplement_ig[names[i]]/5)*100)
       print((stable_ig[names[i]]/5)*100)
       print("--------------------------")

   print("DONE !!!")
