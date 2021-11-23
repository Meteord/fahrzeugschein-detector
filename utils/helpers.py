import cv2
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
from dataclasses import make_dataclass
from utils.constants import SX, SY, EX, EY, CX, CY, TXT, BLACKEN_AROUND, BOX_COLOR, BLACKEN_TEXT_KEYWORDS

#represents information for one bounding box
BoxInfo = make_dataclass("BoxInfo",
                         [(SX, int), #start X
                          (SY, int), #start Y
                          (EX, int), #end X
                          (EY, int), #end Y
                          (CX, float), #center X
                          (CY, float), #center Y
                          (TXT, str),  # containing text
                          (BLACKEN_AROUND, bool)]) # key feature to blacken


def find_nearest(points, point):
    """
    find nearest points
    :param points:  np.array of points of shape (k,2)
    :param point: point of shape (2)
    :return: sorted list of indexes corresponding to the points array, nearer points are at the start of the list
    """
    distances = np.sqrt(np.sum((np.abs(points - point)) ** 2, axis=1))
    sorted_distances = np.argsort(distances)
    return sorted_distances


def decode_predictions(scores, geometry, min_confidence):
    """
    decodes predictions of the text box predictor
    :return: a tuple of the bounding boxes and associated confidences
    """
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


def boxes_to_text(orig, boxes, padding, min_keyword_similarity):
    """
    Extracts the text contained in the boxes
    :param orig: original image
    :param boxes: np.array of shape (L, 4). Representing the bounding boxes containing text
    :param padding: apply extra padding sorrounding the boxes for better results
    :param min_keyword_similarity: between 0 and 100. checks if text contains keyword. if a keyword is with this similarity is present, a box will be blackened
    :return: (output, results): outputs is the output image with bounding boxes, results is [BoxInfo], containing information about text in the boxes
    """
    results: [BoxInfo] = []
    output = orig.copy()
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        if endX < startX:
            endX, startX = startX, endX
        if endY < startY:
            endY, startY = startY, endY

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(orig.shape[1], endX + (dX * 2))
        endY = min(orig.shape[0], endY + (dY * 2))

        startX = max(0, startX - 10)
        startY = max(0, startY - 5)
        endX = endX + 10
        endY = endY + 10

        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ("-l deu --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        centerX = startX + 0.5 * roi.shape[1];
        centerY = startY + 0.5 * roi.shape[0];
        # see if some keyword is present
        containsKeyword = lambda keyword: fuzz.partial_ratio(text.lower(), keyword) > min_keyword_similarity
        containsKeyFeature = any(map(containsKeyword, BLACKEN_TEXT_KEYWORDS))

        # draw the bounding box on the image
        cv2.rectangle(output, (startX, startY), (endX, endY), BOX_COLOR, 2)
        results.append(BoxInfo(startX, startY, endX, endY, centerX, centerY, text, containsKeyFeature))
    return output, results
