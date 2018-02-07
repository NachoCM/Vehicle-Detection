from sklearn.base import BaseEstimator, ClassifierMixin
import inspect

import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


class CarClassifier(BaseEstimator, ClassifierMixin):

    def get_hog_features(self, img, individual_channels=False):
        feature_vec = not individual_channels
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(hog(img[:, :, channel], orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                        cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=False,
                                        visualise=False, feature_vector=feature_vec, block_norm='L2-Hys'))
            if not individual_channels:
                hog_features = np.ravel(hog_features)
        else:
            hog_features = hog(img, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                               cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=False,
                               visualise=False, feature_vector=feature_vec, block_norm='L2-Hys')
        return hog_features

    def get_spatial_features(self, img):
        features = cv2.resize(img, self.spatial_size).ravel()
        return features

    def get_hist_features(self, img):
        channel1_hist = np.histogram(img[:, :, 0], bins=self.hist_bins)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.hist_bins)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.hist_bins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def convert_color(self,img):
        if self.colorspace != 'RGB':
            if self.colorspace == 'HSV':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.colorspace == 'LUV':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.colorspace == 'HLS':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.colorspace == 'YUV':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.colorspace == 'YCrCb':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            converted_image = np.copy(img)
        return converted_image

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs):
       # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for img in imgs:
            # Read in each one by one
            if img.dtype.kind in {'U', 'S'}:
                img = mpimg.imread(img)

            # apply color conversion if other than 'RGB'
            colorspace_image=self.convert_color(img)
            img_features = [self.get_hog_features(colorspace_image),
                            self.get_spatial_features(colorspace_image),
                            self.get_hist_features(colorspace_image)]
            features.append(np.concatenate(img_features))
        # Return list of feature vectors
        return features

    def __init__(self, colorspace='HLS', hog_channel=2, orient=9, pix_per_cell=8, cell_per_block=2,
                 spatial_size=(32, 32), hist_bins=64):
        self.classifier = LinearSVC()

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y=None):

        features = self.extract_features(X)

        self.X_scaler = StandardScaler().fit(np.array(features).astype(np.float64))
        scaled_X = self.X_scaler.transform(np.array(features).astype(np.float64))

        return self.classifier.fit(scaled_X, y)

    def predict(self, X, y=None):
        try:
            getattr(self, "classifier")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        features = self.extract_features(X)
        scaled_X = self.X_scaler.transform(np.array(features).astype(np.float64))
        return self.classifier.predict(scaled_X)

    def score(self, X, y=None):
        features = self.extract_features(X)
        scaled_X = self.X_scaler.transform(np.array(features).astype(np.float64))

        return self.classifier.score(scaled_X, y)

    def get_car_boxes(self, img,xstart,xstop, ystart, ystop, scale, step=2):
        if np.max(img) > 1:
            img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, xstart:xstop, :]
        img_tosearch = self.convert_color(img_tosearch)
        imshape = img_tosearch.shape
        img_scaled = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        full_hog_features = self.get_hog_features(img_scaled, individual_channels=True)

        # Define blocks and steps as above
        nxblocks = (img_scaled.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (img_scaled.shape[0] // self.pix_per_cell) - self.cell_per_block + 1

        # Training images were 64x64px
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = step  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        boxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                #HOG features for the patch
                hog_feat1 = full_hog_features[0][ypos:ypos + nblocks_per_window,xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = full_hog_features[1][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = full_hog_features[2][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                subimg = cv2.resize(img_scaled[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Scale features and make a prediction
                test_features = np.concatenate([hog_features,
                            self.get_spatial_features(subimg),
                            self.get_hist_features(subimg)])

                scaled_features=self.X_scaler.transform(test_features.reshape(1, -1).astype(np.float64))
                prediction = self.classifier.predict(scaled_features)

                if prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    boxes.append(
                        ((xbox_left + xstart, ytop_draw + ystart), (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))

        return np.array(boxes).reshape(-1, 2, 2)

