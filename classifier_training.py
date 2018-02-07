import numpy as np
import glob
from sklearn.model_selection import train_test_split
from car_classifier import CarClassifier
import pickle

# Load cars and notcar images
cars = glob.glob('vehicles/*/*.png')
notcars = glob.glob('non-vehicles/*/*.png')

X = np.hstack((cars, notcars))
# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

hog_class = CarClassifier(colorspace='YCrCb', hog_channel='ALL', pix_per_cell=8, cell_per_block=2,
                          orient=8, spatial_size=(32, 32), hist_bins=32)
hog_class.fit(X_train,y_train)

print(hog_class.score(X_test,y_test))
#Accuracy 0,985

dist_pickle = {}
dist_pickle["classifier"] = hog_class
pickle.dump( dist_pickle, open( "car_classifier.p", "wb" ) )
