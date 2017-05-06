import cv2
from os.path import exists, isdir, basename, join, splitext
import numpy
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
from glob import glob
import scipy.cluster.vq as vq
import cPickle
import pandas as pd

datasetpath = './images'
PRE_ALLOCATION_BUFFER = 100  # for sift
NUMBER_OF_DESCRIPTOR = 20  # for sift
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
CODEBOOK_FILE = 'codebook.file'
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
HISTOGRAMS_FILE = 'trainingdata.svm'


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


def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
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


def extractSift(input_files):
    print "extracting Sift features"
    all_features_dict = {}
    for i, fname in enumerate(input_files):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT(NUMBER_OF_DESCRIPTOR)
        kp, des = sift.detectAndCompute(gray, None)
        print "calculating sift features for: ", fname, "Descriptor :", str(des.shape)
        all_features_dict[fname] = des
    return all_features_dict


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words

def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print 'nclusters have been reduced to ' + str(nwords)
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)



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
    for cat, label in zip(cats, range(ncats)):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_features = extractSift(cat_files)
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label

    print "---------------------"
    print "## computing the visual words via k-means"
    all_features_array = dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]
    nclusters = int(sqrt(nfeatures))
    codebook, distortion = vq.kmeans(all_features_array,
                                     nclusters,
                                     thresh=K_THRESH)
    print "k-Means terminated. Number of cluster: <codebook.shape> ", codebook.shape[0]
    with open(datasetpath + CODEBOOK_FILE, 'wb') as f:
        # save codebook into a binary file
        cPickle.dump(codebook, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print "## compute the visual words histograms for each image"
    print "Number of cluster: <ncluster>",nclusters
    all_word_histgrams = {}
    columns=range(0,nclusters)
    columns.append('file_name')
    columns.append('flower_name')
    df = pd.DataFrame(columns=columns)

    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        #create dataframe to store histogram of visual word occurences
        print type(word_histgram)
        # Convert feature vector to a list
        result = numpy.squeeze(word_histgram).tolist()
        result.append(imagefname)
        label=all_files_labels[imagefname]
        flower_name=cat_label[label]
        result.append(flower_name)
        all_word_histgrams[imagefname] = word_histgram

    print "---------------------"
    print "## write the histograms to file to pass it to the svm"
    writeHistogramsToFile(nclusters,
                          all_files_labels,
                          all_files,
                          all_word_histgrams,
                          datasetpath + HISTOGRAMS_FILE)

    print "---------------------"
    print "## train svm"

