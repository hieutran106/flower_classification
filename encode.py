import cv2
from os.path import exists, isdir, basename, join, splitext
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
from glob import glob
import cPickle

datasetpath = './images'
PRE_ALLOCATION_BUFFER = 100  # for sift
NUMBER_OF_DESCRIPTOR = 20  # for sift
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
CODEBOOK_FILE = 'codebook.file'
HISTOGRAMS_FILE = 'trainingdata.svm'


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
    print "----------------------"
    print "Read codebook from file"
    with open(datasetpath + CODEBOOK_FILE, 'rb') as f:
        codebook = cPickle.load(f)


    print "## compute the visual words histograms for each image"
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
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