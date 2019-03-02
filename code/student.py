import numpy as np
import os
import matplotlib
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist


def get_tiny_images(image_paths):
    '''
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.

    To build a tiny image feature, resize the original image to a very small
    square resolution (e.g. 16x16). You can either resize the images to square
    while ignoring their aspect ratio, or you can crop the images into squares
    first and then resize evenly. Normalizing these tiny images will increase
    performance modestly.

    As you may recall from class, naively downsizing an image can cause
    aliasing artifacts that may throw off your comparisons. See the docs for
    skimage.transform.resize for details:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

    Suggested functions: skimage.transform.resize, skimage.color.rgb2grey,
                         skimage.io.imread, np.reshape
    '''

    n = len(image_paths)
    d = 16**2
    
    retarray = np.zeros((n,d))
    
    for i in range(n):
        retarray[i] = np.array(resize(imread(image_paths[i], as_gray=True),(16,16))).flatten()

    return retarray

def build_vocabulary(image_paths, vocab_size):
    '''
    This function should sample HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    You'll need to generate HOG features using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

    However, the documentation is a bit confusing, so we will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let's say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a really nifty numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).

    The number of feature vectors that come from this reshape is dependent on
    the size of the image you give to hog(). It will fit as many blocks as it
    can on the image. You can choose to resize (or crop) each image to a consistent size
    (therefore creating the same number of feature vectors per image), or you
    can find feature vectors in the original sized image.

    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    to do this. Note that this can take a VERY LONG TIME to complete (upwards
    of ten minutes for large numbers of features and large max_iter), so set
    the max_iter argument to something low (we used 100) and be patient. You
    may also find success setting the "tol" argument (see documentation for
    details)
    '''
    #initialize all features#
    if not os.path.isfile('features.npy'):
        print("No features already extracted.")
        print("Now, extracting features")
        z = 4
        
        image = imread(image_paths[0], as_gray=True)
        features = hog(image, cells_per_block=(z,z),pixels_per_cell = (16,16),feature_vector=True)
    
        for i in range(1, len(image_paths)):
            image = imread(image_paths[i], as_gray=True)
            feat = hog(image, cells_per_block=(z,z),pixels_per_cell = (16,16),feature_vector=True)
            features = np.concatenate((features,feat), axis=0)
        features = features.reshape(-1, z*z*9)
        np.save('features.npy', features)
        print("Done extracting features")
    
    features = np.load('features.npy')
    l_f = len(features)
    
    print("Loaded all features")
    
    print("Start to load or initialize k centers")
    if not os.path.isfile('incenters.npy'):
        print("No previous saved initialized centers")
        print("Now, initializing k centers")
        #initialize k centers#
        centers_ind = set()
        
        if l_f<vocab_size:
            print("error, vocab_size too large")
            return np.array([])
    
        centers_ind.add(np.random.randint(l_f))
        count_center = 1
        for i in range(1,vocab_size):
            sq_dists = np.square(cdist(features, features[list(centers_ind)], 'euclidean'))
            sum_dists = np.sum(sq_dists,axis = 1)
            dis = sum_dists/np.sum(sum_dists)
            ind = np.random.choice(l_f, p = dis)
            while ind in centers_ind:
                ind = np.random.choice(l_f, p = dis)
            centers_ind.add(ind)
            count_center+=1
            if i % 25 ==0:
                print("current progress:", i)
        centers = features[list(centers_ind)]
        np.save('incenters.npy',centers)
        print("Done initializing k centers")
    
    centers = np.load('incenters.npy')
    print("Done loading initial centers")

    print("Start to do k-clustering")
    #k clustering
    old_assignments = np.zeros(len(features))-1
    
    safe_counter = 0
    while True:
        if safe_counter >100:
            break
        clustering = [[] for i in range(vocab_size)]
        
        
        x_dists = cdist(features, centers, 'euclidean')
        new_assignments = np.zeros(l_f)
        for j in range(len(x_dists)):
            ind = np.argmin(x_dists[j])
            new_assignments[j] = ind
            clustering[ind].append(features[j])
            
        centers = []
        for i in range(vocab_size):
            centers.append(np.mean(clustering[i],axis = 0))
        
        if np.sum(new_assignments==old_assignments) >= (0.999*l_f):
            break
        if safe_counter % 5==0:
            print("k-clustering progress:", safe_counter)
        safe_counter += 1
        old_assignments = new_assignments
    print("Done k-clustering")
    print("Return vocabs")
    
    return np.array(centers)

def get_bags_of_words(image_paths):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    '''

    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')

    z = 4
    histograms = []
    #find histogram for each image
    for i in range(len(image_paths)):
        image = imread(image_paths[i], as_gray=True)
        feat = hog(image, cells_per_block=(z,z),pixels_per_cell=(16,16),feature_vector=True).reshape(-1, z*z*9)
        dists = cdist(feat, vocab, 'euclidean')
        hist = np.zeros(len(vocab))
        for j in range(len(dists)):
            ind = np.argmin(dists[j])
            hist[ind] += 1
        hist = hist/np.sum(hist)
        histograms.append(hist)

    return np.array(histograms)

def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    using those learned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    We suggest you look at the sklearn.svm module, including the LinearSVC
    class.
    '''
    
    svm = LinearSVC(multi_class='ovr',tol=0.001).fit(train_image_feats,train_labels)
    

    return svm.predict(test_image_feats)

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict the category for every test image by finding
    the training image with most similar features. You will complete the given
    partial implementation of k-nearest-neighbors such that for any arbitrary
    k, your algorithm finds the closest k neighbors and then votes among them
    to find the most common category and returns that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    The simplest implementation of k-nearest-neighbors gives an even vote to
    all k neighbors found - that is, each neighbor in category A counts as one
    vote for category A, and the result returned is equivalent to finding the
    mode of the categories of the k nearest neighbors. A more advanced version
    uses weighted votes where closer matches matter more strongly than far ones.
    This is not required, but may increase performance.

    Be aware that increasing k does not always improve performance - even
    values of k may require tie-breaking which could cause the classifier to
    arbitrarily pick the wrong class in the case of an even split in votes.
    Additionally, past a certain threshold the classifier is considering so
    many neighbors that it may expand beyond the local area of logical matches
    and get so many garbage votes from a different category that it mislabels
    the data. Play around with a few values and see what changes.

    Useful functions:
        scipy.spatial.distance.cdist, np.argsort, scipy.stats.mode
    '''

    k = 1

    # Gets the distance between each test image feature and each train image feature
    # e.g., cdist
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')

    #TODO:
    # 1) Find the k closest features to each test image feature in euclidean space
    # 2) Determine the labels of those k features
    # 3) Pick the most common label from the k
    # 4) Store that label in a list
    m = test_image_feats.shape[0]
    min_dis = []
    for i in range(m):
        min_dis.append(train_labels[np.argmin(distances[i])])
    #print(min_dis)

    return np.array(min_dis)
