import argparse

import cv2
import numpy as np
import pandas as pd
from imutils.object_detection import non_max_suppression

from utils.constants import SX, SY, EX, EY, CX, CY, BLACKEN_AROUND, BLACKEN_COLOR
from utils.helpers import find_nearest, decode_predictions, boxes_to_text

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-east", "--east", type=str,
                help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=960,
                help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=1280,
                help="resized image height (should be multiple of 32)")
ap.add_argument("-p", "--padding", type=float, default=0.1,
                help="amount of padding to add to each border of ROI")
ap.add_argument("-s", "--similarity", type=int, default=60,
                help="min similarity between containing text of a box and keywords, to be blackened")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]
# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)
# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(origH, origW) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])
# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
print("[INFO] Start detecting text boxes")
blob = cv2.dnn.blobFromImage(image, 1.0, (origW, origH),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
print("[INFO] Finished detecting text boxes")

# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry, args["min_confidence"])
boxes = non_max_suppression(np.array(rects), probs=confidences)

# scale the bounding box coordinates based on the respective ratios
boxes[:, 0] = boxes[:, 0] * rW
boxes[:, 1] = boxes[:, 1] * rH
boxes[:, 2] = boxes[:, 2] * rW
boxes[:, 3] = boxes[:, 3] * rH

print("[INFO] Start detecting text inside the boxes")
# initialize the list of results
(output, results) = boxes_to_text(orig, boxes, args["padding"], args["similarity"])

print("[INFO] Finished detecting text inside the boxes")

df = pd.DataFrame(results)
centers = df[[CX, CY]].to_numpy()
# all bounding boxes that should be blacken around
foundFeatures = df[df[BLACKEN_AROUND]][[CX, CY]]

for index, row in foundFeatures.iterrows():
    # find nearest bounding boxes to blacken bounding boxes
    nearest = find_nearest(centers, row.to_numpy())
    # select nearest and draw black rectangle on top
    # TODO instead of using nearest 10 use some kind of radius
    for nearest_index in nearest[0:10]:
        to_blackout_row = df.iloc[[nearest_index]]
        cv2.rectangle(output, (to_blackout_row[SX].values[0], to_blackout_row[SY].values[0]),
                      (to_blackout_row[EX].values[0], to_blackout_row[EY].values[0]), BLACKEN_COLOR, -1)

print(results)

cv2.imshow("Text Detection", output)

# show the output image
cv2.waitKey(0)
