
import pickle
from car_classifier import CarClassifier
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

dist_pickle = pickle.load(open("car_classifier.p", "rb"))
classifier = dist_pickle["classifier"]


def intersection(box1,box2):
    if (box1[0][0]>box2[1][0]) | (box2[0][0]>box1[1][0]):
        return ((0,0),(0,0))
    if (box1[0][1]>box2[1][1]) | (box2[0][1]>box1[1][1]):
        return ((0,0),(0,0))
    #coordinates of the intersection
    corner1=(max(box1[0][0],box2[0][0]),max(box1[0][1],box2[0][1]))
    corner2=(min(box1[1][0],box2[1][0]),min(box1[1][1],box2[1][1]))
    return (corner1,corner2)

def box_area(box):
    return (box[1][1]-box[0][1])*(box[1][0]-box[0][0])


def add_heat(heatmap, boxes):
    # Iterate through list of bboxes
    for box in boxes:
        # Add 1 for all pixels inside each box
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    #Boost boxes with any intersection
    for box in boxes:
        if np.max(heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]]) > 1:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 2
    # Return updated heatmap
    return heatmap  

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def get_boxes_from_labels(labels):
    boxes=[]
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        boxes.append(bbox)
    # Return the image
    return np.array(boxes).reshape(-1,2,2)

#Draw boxes on the frame, smoothing with previous boxes in similar positions.
#When a previously detected box is not detected anymore, it is kept for a number of frames to allow 
#for imperfect detections
def draw_boxes(img, boxes):
    global current_shown_boxes
    #Number of frames to keep boxes that are not detected anymore.
    #Disabled with few issues...
    max_age=1
    #Boxes to be drawn on the next frame
    new_shown=[]
    #Indexes of currently drawn boxes that are detected again
    variants_found=[]
    for box in boxes:
        for cur_idx in range(len(current_shown_boxes)):
            cur_box,cur_age = current_shown_boxes[cur_idx]
            #If the box intersects with a box currently on the frame:
            if box_area(intersection(cur_box,box))>0:
                #The currently drawn box is moved in the next frame in the direction of the new detection.
                box=(np.array(box)*0.2+np.array(cur_box)*0.8).astype(np.int)
                variants_found.append(cur_idx)
        new_shown.append((box,1))
            
    #Increase age of currently drawn boxes that are not detected anymore.
    #Keep those that are still recent for the next frame
    # for cur_idx in range(len(current_shown_boxes)):
    #     cur_box,cur_age = current_shown_boxes[cur_idx]
    #     if not cur_idx in variants_found:
    #         cur_age+=1
    #         if cur_age<max_age:
    #             new_shown.append((cur_box,cur_age))
    
    #Draw the boxes
    for idx in range(len(new_shown)): 
        box,_ = new_shown[idx]
        cv2.rectangle(img, (box[0][0],box[0][1]), (box[1][0],box[1][1]), (0, 0, 255), 6)
    
    current_shown_boxes=new_shown
    # Return the image
    return img

def get_labels(boxes,threshold=5):
    global persistent_heatmap
    heat = np.zeros((720,1280)).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, boxes)
    
    heat = apply_threshold(heat, threshold)
    
    # Smoothed heatmap
    persistent_heatmap=0.9*persistent_heatmap+0.1*heat
    local_copy=np.copy(persistent_heatmap)
    # Apply threshold to help remove false positives/faded pixels
    local_copy=apply_threshold(local_copy,threshold)
    
    # Find final boxes from smoothed heatmap using label function
    labels = label(local_copy)
    return labels


def find_new_boxes(img):
    boxes2 = classifier.get_car_boxes(img, xstart=0, xstop=1280, ystart=384, ystop=576, scale=2, step=2)
    boxes1 = classifier.get_car_boxes(img, xstart=288, xstop=992, ystart=400, ystop=496, scale=1, step=2)
    boxes = np.concatenate((boxes2, boxes1))

    return boxes


def initialize_processing():
    global current_boxes,persistent_heatmap,current_shown_boxes
    current_boxes=np.array([]).reshape(-1,2,2)
    current_shown_boxes=[] 
    persistent_heatmap=np.zeros((720,1280)).astype(np.float64)


def process_image(img):
    global current_boxes
    new_boxes=find_new_boxes(img)
    labels=get_labels(new_boxes,2)
    current_boxes=get_boxes_from_labels(labels)
    if len(current_boxes)>0:
        draw_boxes(img,current_boxes)
    return img


initialize_processing()
frame_number=1
output_video = 'output_videos/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)