from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
import sklearn.svm
from sklearn.svm import SVC
import math

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
    
def getData2():
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
    return X, y, n_classes

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
    plt.show()

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
    
##------------------------------------------------------      
        
            
def getAverageFace(X):
    (number_of_faces, number_of_features) = X.shape
    tot = np.zeros(number_of_features)
    for face in range(number_of_faces):
        face = X[face]
        tot = (tot + face)/number_of_faces
    average_face = tot
    return average_face
    
def getAverageFaceForEachClass(X, y, nclasses):
    (number_of_faces, number_of_features) = X.shape
    class_to_vec_img = []
    counts_for_each_class = []
    for i in range(nclasses):
        counts_for_each_class.append(0)
        class_to_vec_img.append(np.zeros(number_of_features))
    
    for i in range(number_of_faces):
        current_class = y[i]
        current_img_vec = X[i]
        class_to_vec_img[current_class] = class_to_vec_img[current_class] + current_img_vec
        counts_for_each_class[current_class] += 1
    
    for current_class in range(nclasses):
        class_to_vec_img[current_class] = class_to_vec_img[current_class] / counts_for_each_class[current_class]
        
    return class_to_vec_img
        
def displayAllFacesForAllClasses():
    X,y, nclasses = getData2()
    class_to_vec_img = getAverageFaceForEachClass(X, y, nclasses)
    for current_class in range(nclasses):
        current_face = class_to_vec_img[current_class]
        showIm(current_face)

def displayAverageFaceForEveryone():
    X,y = getData()
    average_face = getAverageFace(X)
    showIm(average_face)

##-------------------

#returns Z = XU and the eigenvectors that were selected
#k - number of raw data to project
#l - number of eigenvectors to use for projection
def getProjectedFaces(X, l, s = False):
    U, mu = PCA(X, s)
    U_effective = extractToplPCA(U, l)
    Z = np.dot(X, U_effective)
    return Z, U_effective
    
def getReconstructedFaces(Z, U):
    if len(Z.shape) == 1 or len(U.shape) == 1:
        Z = Z.reshape(-1,1)
        U = U.reshape(-1,1)
    X_reconstructed = np.dot(Z, U.T)
    return X_reconstructed
    
def extractToplPCA(U, l):
    U_effective = U[:,0]
    if l == 1:
        return U_effective
    for i in range(1,l):
        u = U[:,i]
        U_effective = np.column_stack((U_effective, u))
    return U_effective
    
def displayFaces(X):
    (number_of_faces, number_of_features) = X.shape
    for i_f in range(number_of_faces):
        current_face = X[i_f]
        showIm(current_face)
        
def displayReconstructedFaces(X_reconstructed):
    plotGallery([vecToImage(X_reconstructed[i,:]) for i in range(12)])
    plt.show()
    
def part4(l_input):
    (X, y, nclasses) = getData2()

    Z, U_effective = getProjectedFaces(X, l = l_input) #Z = XU

    X_reconstructed = getReconstructedFaces(Z, U_effective) #X' = ZUt = XUUt

    displayReconstructedFaces(X_reconstructed)
               
##----
def getNewY(y, discriminate_person = 4):
    new_y = np.zeros(len(y))
    for i in range(len(y)):
        current_y = y[i]
        if current_y == discriminate_person:
            new_y[i] = 1
        else:
            new_y[i] = -1
    return new_y
            
def getTrainTestData(X, newY, testSize = 0.75):
    (trainX, testX, trainY, testY) = sklearn.cross_validation.train_test_split(X, newY, test_size = testSize)
    return (trainX, testX, trainY, testY)
    
#Part5
def part5():
    (X, y, nclasses) = getData2()
    Z, U_effective = getProjectedFaces(X, l = 100, s = True)
    newY = getNewY(y)
    (trainX, testX, trainY, testY) = getTrainTestData(Z, newY, testSize = 0.75)
    errors_train = []
    errors_test = []
    logc_list = []
    for logc in range(-10, 10, 1):
        c = math.pow(10, logc)
        clf = sklearn.svm.SVC(kernel = 'linear', C = c)
        clf.fit(trainX, trainY)
        error_test = 1 - clf.score(testX, testY)
        error_train = 1 - clf.score(trainX, trainY)
        logc_list.append(logc)
        errors_train.append(error_train)
        errors_test.append(error_test)
    plt.plot(np.array(logc_list), np.array(errors_train))
    plt.plot(np.array(logc_list), np.array(errors_test))
    plt.show()
    
#Part6
def part6():
    (X, y, nclasses) = getData2()
    newY = getNewY(y)
    c = 100
    errors_train = []
    errors_test = []
    l_list = []
    clf = sklearn.svm.SVC(kernel = 'linear', C = c)
    for current_l in range(1, 301, 10):
        Z, U_effective = getProjectedFaces(X, l = current_l, s = True)
        (trainX, testX, trainY, testY) = sklearn.cross_validation.train_test_split(X, newY, test_size = 0.75)
        clf.fit(trainX, trainY)
        
        error_test = 1 - clf.score(testX, testY)
        error_train = 1 - clf.score(trainX, trainY)
        errors_train.append(error_train)
        errors_test.append(error_test)
        l_list.append(current_l)
    plt.plot(np.array(l_list), np.array(errors_train))
    plt.plot(np.array(l_list), np.array(errors_test))
    plt.show()
    
#----

#X - an n x d numpy array of n data points and d features
#init - a k x d numpy array of k data points, each with d features (which are the initial guesses for the centroids)
#---Returns---
# centroids - is a k x d numpy array of k data points, idicating the final centroids
#clusterAssignments - is numpy array of n integers, each in the range from 0 to k-1 indicating
#the cluster they are assiged to
def ml_k_means(X, init):
    d = X.shape[1]
    k = init.shape[0]
    n = X.shape[0]
    clusterAssignments = np.zeros(n)
    centroids = init
    converge = False
    convergence_count = 0
    while not converge:
        convergence_count = 0
        #for fixed centroids z, assign best clusters C
        cost_before_update = computeTotalCost(centroids, clusterAssignments, X)
        for i in range(n):
            x_i = X[i]
            smallest_d = float("inf")
            for j in range(k):
                z_j = centroids[j]
                d = dist(z_j, x_i)
                if smallest_d > d:
                    smallest_d = d
                    closest_z_index = j
            clusterAssignments[i] = closest_z_index
            
        current_tot_cost = computeTotalCost(centroids, clusterAssignments, X)
        if cost_before_update == current_tot_cost:
            convergence_count += 1
        
        #for fixed clusters, assign best centroids
        cost_before_update = current_tot_cost
        new_cluster = np.zeros(k, d) #return k by d
        count_of_points_in_clusters = np.zeros(k)
        for i in range(n):
            x_i = X[i]
            cluster_index = clusterAssignments[i]
            count_of_points_in_clusters[cluster_index] += 1
            new_cluster[cluster_index] = new_cluster[cluster_index] + x_i
        for j in range(k):
            size_of_cluster = count_of_points_in_clusters[j]
            new_cluster[j] = new_cluster[j]/size_of_cluster
            
        centroids = new_cluster
        current_tot_cost = computeTotalCost(centroids, clusterAssignments, X)
        if cost_before_update == current_tot_cost:
            convergence_count += 1
            
        if convergence_count == 2:
            convergence_count = True
            break
            
    return (centroids, clusterAssignments)

#eucledian distance
def dist(a, b):
    c = np.linalg.norm(a-b)
    return c * c
  
#X - an n x d numpy array of n data points and d features
#init - a k x d numpy array of k data points, each with d features (which are the initial guesses for the centroids)
#---Returns--- 
#total cost
def computeTotalCost(centroids, clusterAssignments, X):
    n = centroids.shape[0]
    total_cost = 0
    for i in range(n):
        x_i = X[i]
        cluster_index = clusterAssignments[i]
        current_centroid = centroids[cluster_index]
        d = dist(x_i, current_centroid)
        total_cost += d
    return total_cost
    
#plt.plot([0.5,0.5], [0,1] , 'ro')
#plt.show()

#x - n by 2
def plotRawData(X, axis):
    rows = X.shape[0]
    x_coord = []
    y_coord = []
    for row in range(rows):
        x = X[row][0]
        y = X[row][1]
        x_coord.append(x)
        y_coord.append(y)
    plt.plot(x_coord, y_coord , 'ro')
    plt.axis(axis)
    plt.show()

def plotCluster(clusterAssignments, X):
    n = X.shape[0]
    clusters_xcoords = [[],[],[]]
    clusters_ycoords = [[],[],[]]
    #x_max = float("-inf")
    #y_max = float("-inf")
    #x_min = float("inf")
    #y_min = float("inf")
    print n
    for i in range(n):
        x_i = X[i]
        print type(x_i[0])
        print "xcords",clusters_xcoords
        print "ycords",clusters_ycoords
        #x_max = max(x_max, x_i[0])
        #y_max = max(y_max, x_i[0])
        #x_min = min(x_min, x_i[1])
        #y_min = min(y_min, x_i[1])
        cluster_index = clusterAssignments[i]
        clusters_xcoords[cluster_index].append(float(x_i[0]))
        clusters_ycoords[cluster_index].append(float(x_i[1]))
    print clusters_xcoords
    print clusters_ycoords
    print clusters_xcoords[0]
    print clusters_ycoords[0]
    print clusters_xcoords[1]
    print clusters_ycoords[1]
    #plt.plot(clusters_xcoords[0], clusters_ycoords[0], 'ro', clusters_xcoords[1], clusters_ycoords[1], 'bo', clusters_xcoords[2], clusters_ycoords[2], 'go')
    #plt.plot(clusters_xcoords[0], clusters_ycoords[0], 'ro')
    plt.plot([0.1,0.4], [0.1, 0.4] , 'ro', [0.1,0.2], [0.1,0.2] , 'bo', [0.3,0.5], [0.4, 0.5], 'go')
    
    #plt.axis([x_min, y_min, x_max, y_max])
    #plt.axis([0, 6, 0, 20])
    plt.axis([0, 0, 1, 1])
    plt.ylabel("y")
    plt.show()
        
X = np.array( [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4] ] )
clusters = np.array([0,1 ,2 , 0])
plotCluster(clusters, X)  
#plt.plot([0.1,0.4], [0.1, 0.4] , 'ro', [0.1,0.2], [0.1,0.2] , 'bo', [0.3,0.5], [0.4, 0.5], 'go')
#plt.axis([-1, 3, -1, 3])
plt.show()
    
##--------------------------------------------------------------------------

print "ML 6.036 code running"

# X: an n x d array, in which each row represents an image
# y: a 1 x n vector, elements of which are integers between 0 and nc-1
#    where nc is the number of classes represented in the data

#(X, y, nclasses) = getData2()
#Z, U_effective = getProjectedFaces(X, l = 100, s = True)
#X_reconstructed = getReconstructedFaces(Z, U_effective)
#newY = getNewY(y)
#(trainX, testX, trainY, testY) = getTrainTestData(X_reconstructed, newY, testSize = 0.75)
#clf = sklearn.svm.SVC(kernel = 'linear', C = 1.0)
#clf.fit(trainX, trainY)
#score = clf.score(testX,testY)

#part4(1)
#part5()
#part6()


print "End of ML 6.036 code running"