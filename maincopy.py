import cv2
import numpy as np

# Image input
img = cv2.imread('Images/cabletryred3.jpg')
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

# Resize for large images
imgHeight, imgWidth, _ = img.shape
if(imgHeight > 5000 and imgWidth > 7000):
    imgCopy = rescaleFrame(imgCopy,0.4)

# Red color findings
redhsv_image = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2HSV)
imgRedBlur = cv2.GaussianBlur(redhsv_image, (5, 5), 1)
lower_red = np.array([155, 25, 0])
upper_red = np.array([179, 255, 255])
redmask = cv2.inRange(redhsv_image, lower_red, upper_red)
cv2.imshow('redmask', redmask)

# Contour output
contours, heirarchy = cv2.findContours(redmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# White color findings
hsv_image = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 100])
upper_white = np.array([255, 30, 255])
mask = cv2.inRange(hsv_image, lower_white, upper_white)
cv2.imshow('mask', mask)

bounding_rects = []

#for white space detection
def is_white_space(mask, x, y, width, height):
    for i in range(y, y + height):
        for j in range(x, x + width):
            if mask[i, j] != 255:
                return False
    return True

# loop for contours
for cnt in contours:
    area = cv2.contourArea(cnt)
    print(area, 'area')
    if area > 10:
        x, y, w, h = cv2.boundingRect(cnt)
        duplicate = False
        for rect in bounding_rects:
            if abs(rect[0] - x) < 10 and abs(rect[1] - y) < 10 and abs(rect[2] - w) < 10 and abs(rect[3] - h) < 10:
                duplicate = True
                break

            # If not a duplicate, draw the rectangle and add to the list
        if not duplicate:
            cv2.rectangle(imgCopy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            bounding_rects.append((x, y, w, h))
            try:
                if w > anno_width:
                    start_x = x
                    end_x = x + w
                    while start_x < end_x:
                        if is_white_space(mask, start_x-anno_width//2, y-anno_height, anno_width, anno_height):
                            cv2.rectangle(imgCopy, (start_x-anno_width//2, y-anno_height),
                                    (start_x + anno_width//2, y), (255, 0, 0), 2)
                            break
                        else:
                            if is_white_space(mask, start_x-anno_width//2, y+h, anno_width, anno_height):
                                cv2.rectangle(imgCopy, (start_x-anno_width//2, y+h),(start_x + anno_width//2, y+h+anno_height), (255, 0, 0), 2)
                                break
                            else:
                                start_x = start_x + 1
                   
                elif h > anno_height:
                    start_y = y
                    end_y = y + h
                    while start_y < end_y:
                        if is_white_space(mask, x-anno_width, start_y-anno_height, anno_width, anno_height):
                            cv2.rectangle(imgCopy, (x-anno_width, start_y-anno_height//2),
                                    (x, start_y+anno_height//2), (255, 255, 0), 2)
                            break
                        else:
                            if is_white_space(mask, x+w, start_y-anno_height//2, anno_width, anno_height):
                                cv2.rectangle(imgCopy, (x+w, start_y-anno_height//2),(x+w+anno_width, start_y+anno_height//2), (255, 255, 0), 2)
                                break
                            else:
                                start_y = start_y + 1

                else:
                    if is_white_space(mask, x-anno_width,y-anno_height, anno_width, anno_height) :
                        cv2.rectangle(imgCopy, (x, y), (x - anno_width, y - anno_height),
                                (255, 0, 255), 2)
                        break
                    else:
                        if is_white_space(mask, x+w, y+h, anno_width, anno_height) :
                            cv2.rectangle(imgCopy, (x+w, y+h), (x+w+anno_width, y +h+anno_height),(255, 0, 255), 2)
                        else :
                            print ("Not enough space")

            except IndexError:
                print("Error")
                pass
                
# Output image
cv2.imshow('Output', imgCopy)
cv2.waitKey(0)