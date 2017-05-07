import cv2
from os.path import exists, isdir, basename, join, splitext
import numpy
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
from glob import glob
import scipy.cluster.vq as vq
import cPickle
import pandas as pd

datasetpath = './images'
PRE_ALLOCATION_BUFFER = 300  # for sift
NUMBER_OF_DESCRIPTOR = 300  # for sift
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
CODEBOOK_FILE = 'codebook.file'
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
HISTOGRAMS_FILE = 'trainingdata_svm.csv'


def get_categories(datasetpath):
    cat_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname))
                      for fname in glob(path + "/*")
                      if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files

def dict2numpy(dict, total_features):
    array = zeros((total_features, 128),dtype=numpy.float32)
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 128))
    return array

def dict2numpy_old(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128),dtype=numpy.float16)
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 128))
    return array


def extractSift(input_files, mask=False):
    all_features_dict = {}
    num_descriptor = 0
    for i, fname in enumerate(input_files):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #sift = cv2.SIFT(NUMBER_OF_DESCRIPTOR)
        sift = cv2.SIFT(contrastThreshold=0.06)
        if (mask):
            (h, w) = img.shape[:2]
            (cX, cY) = ((int)(w * 0.5), (int)(h * 0.5))
            (axesX, axesY) = ((int)((w * 0.8) / 2), (int)((h * 0.8) / 2))
            ellipMask = numpy.zeros(img.shape[:2], dtype="uint8")
            cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
            kp, des = sift.detectAndCompute(gray, ellipMask, None)
        else:
            kp, des = sift.detectAndCompute(gray, None)

        if (len(kp)==0):
            print "No feature found for image :",fname
        else:
            num_descriptor += len(des)
        all_features_dict[fname] = des
    return all_features_dict, num_descriptor


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words


if __name__ == '__main__':

    print "---------------------"
    print "## loading the images and extracting the sift features"
    cats = get_categories(datasetpath)
    ncats = len(cats)
    print "searching for folders at " + datasetpath
    print "found following folders / categories:"
    print cats
    print "---------------------"
    all_files = []
    all_files_labels = {}
    all_features = {}
    cat_label = {}
    cat_dic = {}
    total_descriptor = 0
    for cat, label in zip(cats, range(ncats)):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_dic[cat] = cat_files
        cat_features, nDes = extractSift(cat_files, mask=True)
        print "Features for images in category \"", cat, "\":", str(nDes)
        total_descriptor += nDes
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label

    print "Total features extracted:", str(total_descriptor)
    print "---------------------"
    print "## computing the visual words via k-means"
    all_features_array = dict2numpy(all_features, total_descriptor)
    nfeatures = all_features_array.shape[0]
    nclusters = int(sqrt(nfeatures / 2))
    nclusters=100
    print "Number of cluster:", nclusters
    codebook, distortion = vq.kmeans(all_features_array,
                                     nclusters,
                                     thresh=K_THRESH)
    print "k-Means terminated."
    with open(datasetpath + CODEBOOK_FILE, 'wb') as f:
        # save codebook into a binary file
        cPickle.dump(codebook, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print "## compute the visual words histograms for each image"
    print "Number of cluster: <ncluster>", nclusters
    all_word_histgrams = {}
    columns = range(0, nclusters)
    columns.append('file_name')
    columns.append('flower_name')
    df = pd.DataFrame(columns=columns)

    for cat in cat_dic:
        for imagefname in cat_dic[cat]:
            word_histgram = computeHistograms(codebook, all_features[imagefname])
            # create dataframe to store histogram of visual word occurences
            # Convert feature vector to a list
            result = numpy.squeeze(word_histgram).tolist()
            result.append(imagefname)
            result.append(cat)
            df.loc[len(df)] = result

    print "---------------------"
    print "## write the histograms to file to pass it to the svm"
    df.to_csv(datasetpath + HISTOGRAMS_FILE, index_label="ID")
    print "---------------------"
    print "## train svm"
