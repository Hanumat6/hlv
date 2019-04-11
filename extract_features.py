from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
"""from config import *"""
pos_im_path = 'F:\\human-detector-master\\data\\images\\pos_person'
neg_im_path = 'F:\\human-detector-master\\data\\images\\neg_person'
min_wdw_sz = [64, 128]
step_size = [10, 10]
orientations = 9
pixels_per_cell = [6, 6]
cells_per_block = [2, 2]
visualize = False
normalize = True
pos_feat_ph = 'F:\\human-detector-master\\data\\features\\pos'
neg_feat_ph = 'F:\\human-detector-master\\data\\features\\neg'
model_path = 'F:\\human-detector-master\\data\\models'
threshold = .3

    
def extract_features():
    des_type = 'HOG'

    
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)

    print ("Calculating the descriptors for the positive samples and saving them")
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        
        
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            #fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd=hog(im, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2),block_norm='L1', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print ("Positive features saved in {}".format(pos_feat_ph))

    print ("Calculating the descriptors for the negative samples and saving them")
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            
            fd=hog(im, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2),block_norm='L1', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)   
        joblib.dump(fd, fd_path)
    print ("Negative features saved in {}".format(neg_feat_ph))

    print ("Completed calculating features from training images")

if __name__=='__main__':
    extract_features()
