import cv2
import numpy as np

def nothing(x):
	pass

def normal_canny(image, lower, upper):
	edged = cv2.Canny(image, lower, upper)
	return edged

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged


# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("/home/nitesh/Downloads/dip.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

cv2.namedWindow("Canny Edge Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Canny Edge Detector', 600,600)

# cv2.createTrackbar("sigma", "Canny Edge Detector", 0, 100, nothing)
cv2.createTrackbar("lower", "Canny Edge Detector", 0, 100, nothing)
cv2.createTrackbar("upper", "Canny Edge Detector", 0, 100, nothing)

while True:
	# sigma = cv2.getTrackbarPos("sigma", "Canny Edge Detector")
	lower = cv2.getTrackbarPos("lower", "Canny Edge Detector")
	upper = cv2.getTrackbarPos("upper", "Canny Edge Detector")

	# edged = auto_canny(gray, sigma)
	edged = normal_canny(blurred, float(lower), float(upper))

	print(edged)

	cv2.imshow("Canny Edge Detector", edged)

	if cv2.waitKey(30)==27:
		break

cv2.destroyAllWindows()
