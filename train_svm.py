
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
import numpy as np

#from config import *

pos_feat_ph = 'F:\\re-svm\\data\\features\\pos'
neg_feat_ph = 'F:\\re-svm\\data\\features\\neg'
model_path = 'F:\\re-svm\\data\\models'




def train_svm():
    
    
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    
    for feat_path in glob.glob(os.path.join(pos_feat_ph,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    
    
    for feat_path in glob.glob(os.path.join(neg_feat_ph,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print (np.array(fds).shape,len(labels))
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print ("Training a Linear SVM Classifier")
        
        clf.fit(fds, labels)
        
        
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
            
        
        
        p=os.path.join(model_path,"svm.model")
        joblib.dump(clf,p)
        print ("Classifier saved to {}".format(model_path))  
        
if __name__=='__main__':
    train_svm()