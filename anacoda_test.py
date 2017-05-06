
from os.path import exists, isdir, basename, join, splitext
from glob import glob

datasetpath = './images'
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]



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



if __name__ == '__main__':

    print("---------------------")
    print("## loading the images and extracting the sift features")
    cats = get_categories(datasetpath)
    ncats = len(cats)
    print("searching for folders at " + datasetpath)
    print("found following folders / categories:")
    print(cats)
    print("---------------------")
    all_files = []
    all_files_labels = {}
    all_features = {}
    cat_label = {}
    need_dict ={}
    for cat, label in zip(cats, range(ncats)):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        need_dict[cat]=cat_files
        #cat_features = extractSift(cat_files)
        all_files = all_files + cat_files
        #all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label