import numpy as np
import cv2

gray = cv2.imread('/home/nitesh/Downloads/dip.jpg')
edges = cv2.Canny(gray,10,200,apertureSize = 3)
cv2.imwrite('edges-50-150.jpg',edges)
cv2.imshow("", edges)
cv2.waitKey(0)
minLineLength=100
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)



a,b,c = lines.shape
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite('houghlines5.jpg',gray)
    cv2.imshow("Gray "+str(i), gray)
    cv2.waitKey(0)
