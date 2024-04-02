import cv2
import numpy as np

# Image input
img = cv2.imread('Images/paint6.jpg')
imgCopy = img.copy()
cv2.imshow('Original', img)

# Annotation size
anno_width = 30
anno_height = 20

# Image resizer
def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


imgHeight, imgWidth, _ = img.shape
if (imgHeight > 3000 and imgWidth > 5000):
    imgCopy = rescaleFrame(imgCopy, 0.8)

# Red color findings
redhsv_image = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2HSV)
lower_red = np.array([155, 25, 0])
upper_red = np.array([179, 255, 255])
redmask = cv2.inRange(redhsv_image, lower_red, upper_red)
cv2.imshow('Red Mask', redmask)

# Contour output
contours, heirarchy = cv2.findContours(
    redmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# White color findings
hsv_image = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 231])  # 0,0,100
upper_white = np.array([180, 18, 255])  # 255,30,255
mask = cv2.inRange(hsv_image, lower_white, upper_white)
cv2.imshow('Mask', mask)

# for white space detection
def is_white_space(mask, x, y, width, height):
    for i in range(y, y + height):
        for j in range(x, x + width):
            if mask[i, j] != 255:
                return False
    return True


bounding_rects = []

# loop for contours
for cnt in contours:
    # compute rotated rectangle (minimum area)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int64(box)

    area = cv2.contourArea(cnt)
    if area > 10:
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]
        x = x1
        y = y1
        w = int(rect[1][0])
        h = int(rect[1][1])
        rotation = rect[2]
        cv2.drawContours(imgCopy, [box], 0, (0, 255, 255), 2)
        bounding_rects.append((x, y, w, h))
        try:
            if x1 == x4 and y1 == y2 and x2 == x3 and y3 == y4:
                    if w > anno_width:
                        start_x = x1
                        end_x = x2 + anno_width
                        while start_x < end_x:
                            if is_white_space(mask, start_x-anno_width, y1-anno_height, anno_width, anno_height):
                                cv2.rectangle(imgCopy, (start_x-anno_width, y1-anno_height),
                                              (start_x, y1), (255, 0, 0), 2)
                                break
                            else:
                                if is_white_space(mask, start_x-anno_width, y1+h, anno_width, anno_height):
                                    cv2.rectangle(imgCopy, (start_x-anno_width, y1+h),
                                                  (start_x , y+h+anno_height), (255, 0, 0), 2)
                                    break
                                else:
                                    start_x = start_x + 1

                    elif h > anno_height: 
                        start_y = y1
                        end_y = y4 + anno_height
                        while start_y < end_y:
                            if is_white_space(mask, x1-anno_width, start_y-anno_height, anno_width, anno_height):
                                cv2.rectangle(imgCopy, (x1-anno_width, start_y-anno_height),
                                              (x1, start_y), (255, 255, 0), 2)
                                break
                            else:
                                if is_white_space(mask, x1+w, start_y-anno_height, anno_width, anno_height):
                                    cv2.rectangle(imgCopy, (x1+w, start_y-anno_height),
                                                  (x1+w+anno_width, start_y), (255, 255, 0), 2)
                                    break
                                else:
                                    start_y = start_y + 1

                    else:
                        if is_white_space(mask, x1-anno_width, y1-anno_height, anno_width, anno_height):
                            cv2.rectangle(imgCopy, (x1, y1), (x1 - anno_width, y1 - anno_height),
                                          (255, 0, 255), 2)
                            break
                        else:
                            if is_white_space(mask, x1+w, y1+h, anno_width, anno_height):
                                cv2.rectangle(
                                    imgCopy, (x1+w, y1+h), (x1+w+anno_width, y1 + h +anno_height), (255, 0, 255), 2)
                                break
                            else:
                                print("Not enough space")
            else:
                print("Not a rectangle")

        except IndexError:
            print("Error")
            pass

# Output image
cv2.imshow('Output', imgCopy)
cv2.waitKey(0)
