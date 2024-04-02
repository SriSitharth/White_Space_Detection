# imports:
import cv2
import numpy as np

# Read image:
inputImage = cv2.imread('Images/paint.jpg')
# Store a copy for results:
inputCopy = inputImage.copy()

# Convert BGR to grayscale:
grayInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Threshold via Otsu
_, binaryImage = cv2.threshold(grayInput, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Set kernel (structuring element) size:
kernelSize = (9, 9)

# Set operation iterations:
opIterations = 2

# Get the structuring element:
morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)

# Perform Dilate:
dilateImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

# Reduce matrix to a n row x 1 columns matrix:
reducedImage = cv2.reduce(dilateImage, 1, cv2.REDUCE_MAX)

# Invert the reduced image:
reducedImage = 255 - reducedImage

# Find the big contours/blobs on the filtered image:
contours, hierarchy = cv2.findContours(reducedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Store the poly approximation and bound
contoursPoly = [None] * len(contours)
separatingLines = [ ]

# We need some dimensions of the original image:
imageHeight = inputCopy.shape[0]
imageWidth = inputCopy.shape[1]

# Look for the outer bounding boxes:
for i, c in enumerate(contours):

    # Approximate the contour to a polygon:
    contoursPoly = cv2.approxPolyDP(c, 3, True)

    # Convert the polygon to a bounding rectangle:
    boundRect = cv2.boundingRect(contoursPoly)

    # Get the bounding rect's data:
    [x,y,w,h] = boundRect

# Calculate line middle (vertical) coordinate,
# Start point and end point:
lineCenter = y + (0.5 * h)
startPoint = (0,int(lineCenter))
endPoint =  (int(imageWidth),int(lineCenter))

# Store start and end points in list:
separatingLines.append((startPoint, endPoint))

# Draw the line:
color = (0, 255, 0)
cv2.line(inputCopy, startPoint, endPoint, color, 2)

# Show the image:
cv2.imshow("inputCopy", inputCopy)
cv2.waitKey(0)