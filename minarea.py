import cv2
import numpy as np

# Load the image
image_path = "Images/paint.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to create a binary image
_, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour (assuming the rectangle you want to draw is the largest contour)
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Calculate the center of the bounding box
center_x = x + w // 2
center_y = y + h // 2

# Calculate the angle of the bounding box (assuming the angle is aligned with the image's x-axis)
angle = 30  # Replace this with the actual angle if known

# Calculate the points for the 30x20 rectangle with the given angle
rect_width = 30
rect_height = 20

angle_rad = np.deg2rad(angle)
cos_theta = np.cos(angle_rad)
sin_theta = np.sin(angle_rad)

# Calculate the four corners of the rectangle
corner1 = (center_x + rect_width // 2, center_y + rect_height // 2)
corner2 = (center_x - rect_width // 2, center_y + rect_height // 2)
corner3 = (center_x - rect_width // 2, center_y - rect_height // 2)
corner4 = (center_x + rect_width // 2, center_y - rect_height // 2)

# Apply rotation transformation to the corners
rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
rotated_corner1 = np.dot(rot_matrix, corner1 - (center_x, center_y)) + (center_x, center_y)
rotated_corner2 = np.dot(rot_matrix, corner2 - (center_x, center_y)) + (center_x, center_y)
rotated_corner3 = np.dot(rot_matrix, corner3 - (center_x, center_y)) + (center_x, center_y)
rotated_corner4 = np.dot(rot_matrix, corner4 - (center_x, center_y)) + (center_x, center_y)

# Convert the corners to integer coordinates
rect_points = np.array([rotated_corner1, rotated_corner2, rotated_corner3, rotated_corner4], dtype=np.int32)

# Draw the rectangle on the image
cv2.polylines(image, [rect_points], isClosed=True, color=(0, 255, 0), thickness=2)

# Display the image
cv2.imshow("Image with Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
