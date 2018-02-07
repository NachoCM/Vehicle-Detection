import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo.yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, yolo_eval
from yolo.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from moviepy.editor import VideoFileClip




sess = K.get_session()
class_names = read_classes("yolo/model_data/coco_classes.txt")
# Generate colors for drawing bounding boxes.
colors = generate_colors(class_names)
anchors = read_anchors("yolo/model_data/yolo_anchors.txt")
image_shape = (720., 1280.)
#Load pretrained model downloaded from the YOLO website and converted using a function by Allan Zelener.
yolo_model = load_model("yolo/model_data/yolo.h5")
yolo_model.summary()
#Transform the output of yolo_model into a usable format
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
#Filter boxes to finally get bounding boxes for the selected classes, and scores associated with them
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)




def predict(sess, img):

    # Preprocess your image
    image, image_data = preprocess_image(img, model_image_size = (608, 608))

    # Run the session to find bounding boes
    out_scores, out_boxes, out_classes = sess.run((scores,boxes,classes),feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    return np.array(image, dtype='uint8')




def process_image(img):
    return predict(sess,img)


frame_number=1
output_video = 'output_videos/project_video_yolo.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)