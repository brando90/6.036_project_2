from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
import sklearn.svm
from sklearn.svm import SVC

# Returns
# X: an n x d array, in which each row represents an image
# y: a 1 x n vector, elements of which are integers between 0 and nc-1
#    where nc is the number of classes represented in the data

# Warning: this will take a long time the first time you run it.  It
# will download data onto your disk, but then will use the local copy
# thereafter.  
def getData():
    global X, n, d, y, h, w
    lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.4)
    n, h, w = lfw_people.images.shape
    X = lfw_people.data
    d = X.shape[1]
    y = lfw_people.target
    n_classes = lfw_people.target_names.shape[0]
    print("Total dataset size:")
    print("n_samples: %d" % n)
    print("n_features: %d" % d)
    print("n_classes: %d" % n_classes)
    return X, y

# Input
# im: a row or column vector of dimension d
# size: a pair of positive integers (i, j) such that i * j = d
#       defaults to the right value for our images
# Opens a new window and displays the image
lfw_imageSize = (50,37)
def showIm(im, size = lfw_imageSize):
    plt.figure()
    im = im.copy()
    im.resize(*size)
    plt.imshow(im.astype(float), cmap = cm.gray)

# Take an eigenvector and make it into an image
def vecToImage(x, size = lfw_imageSize):
  im = x/np.linalg.norm(x)
  im = im*(256./np.max(im))
  im.resize(*size)
  return im

# Plot an array of images
# Input
# - images: a 12 by d array
# - title: string title for whole window
# - subtitles: a list of 12 strings to be used as subtitles for the
#              subimages, or an empty list
# - h, w, n_row, n_col: can be used for other image sizes or other
#           numbers of images in the gallery

def plotGallery(images, title='plot', subtitles = [],
                 h=50, w=37, n_row=3, n_col=4):
    plt.figure(title,figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(len(images), n_row * n_col)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        if subtitles:
            plt.title(subtitles[i], size=12)
        plt.xticks(())
        plt.yticks(())    
    
# Perform PCA, optionally apply the "sphering" or "whitening" transform, in
# which each eigenvector is scaled by 1/sqrt(lambda) where lambda is
# the associated eigenvalue.  This has the effect of transforming the
# data not just into an axis-aligned ellipse, but into a sphere.  
# Input:
# - X: n by d array representing n d-dimensional data points
# Output:
# - u: d by n array representing n d-dimensional eigenvectors;
#      each column is a unit eigenvector; sorted by eigenvalue
# - mu: 1 by d array representing the mean of the input data
# This version uses SVD for better numerical performance when d >> n

def PCA(X, sphere = False):
    (n, d) = X.shape
    mu = np.mean(X, axis=0)
    (x, l, v) = np.linalg.svd(X-mu)
    l = np.hstack([l, np.zeros(v.shape[0] - l.shape[0], dtype=float)])
    u = np.array([vi/(li if (sphere and li>1.0e-10) else 1.0) \
                  for (li, vi) \
                  in sorted(zip(l, v), reverse=True, key=lambda x: x[0])]).T
    return u, mu

# Selects a subset of images from the large data set.  User can
# specify desired classes and desired number of images per class.
# Input:
# - X: n by d array representing n d-dimensional data points
# - y: 1 by n array reprsenting the integer class labels of the data points
# - classes: a list of integers naming a subset of the classes in the data
# - nim: number of integers desired
# Return:
# - X1: nim * len(classes) by d array of images
# - y1: 1 by nim * len(classes) array of class labels
def limitPics(X, y, classes, nim):
  (n, d) = X.shape
  k = len(classes)
  X1 = np.zeros((k*nim, d), dtype=float)
  y1 = np.zeros(k*nim, dtype = int)
  index = 0
  for ni, i in enumerate(classes):      # for each class
    count = 0                           # count how many samples in class so far
    for j in range(n):                  # look over the data
      if count < nim and y[j] == i:     # element of class
        X1[index] = X[j]
        y1[index] = ni
        index += 1
        count += 1
  return X1, y1

# Provides an initial set of data points to use to initialize
# clustering.  It "cheats" by using the class labels, and picks the
# medoid of each class.
# Input:
# - X: n by d array representing n d-dimensional data points
# - y: 1 by n array representing integer class labels
# - k: number of classes
# Output:
# - init: k by d array representing initial cluster medoids
def cheatInit(X, y, k):
    (n, d) = X.shape
    init = np.zeros((k, d), dtype=float)
    for i in range(k):
        (index, dist) = cheatIndex(X, y, i, l2Sq)
        init[i] = X[index]
    return init

def l2Sq (x,y):
    return np.sum(np.dot((x-y), (x-y).T))

def cheatIndex(X, clusters, j, metric):
    n, d = X.shape
    bestDist = 1.0e10
    index = 0
    for i1 in xrange(n):
        if clusters[i1] == j:
            dist = 0
            C = X[i1,:]
            for i2 in xrange(n):
                if clusters[i2]  == j:
                    dist += metric(C, X[i2,:])
            # print dist
            if dist < bestDist:
                bestDist = dist
                index = i1
    return index, bestDist


# Scores the quality of a clustering, in terms of its agreement with a
# vector of labels
# Input:
# - clustering: (medoids, clusters, indices) of type returned from kMedoids
# - y: 1 by n array representing integer class labels
# Output:
# numerical score between 0 and 1
def scoreMedoids(clustering, y):
    (medoids, mIndex, cluster) = clustering
    n = cluster.shape[0]                  # how many samples
    # The actual label for each medoid, which we associate with
    # samples in cluster
    medoidLabels = np.array([y[i] for i in mIndex]) 
    print medoidLabels
    count = len(set(medoidLabels.tolist())) # how many actual people predicted
    # For each sample, what is the label implied by its cluster
    clusterLabels = np.array([medoidLabels[c] for c in cluster])
    score = sum([1 if y[i]==clusterLabels[i] else 0 \
                 for i in xrange(n)])/float(n)
    return score
