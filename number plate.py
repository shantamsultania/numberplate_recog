import cv2

# cas = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') 
# New version of cv2 does not support this method
cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

image = cv2.imread('download.jpg')

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

plate_de = cas.detectMultiScale(gray,1.1,5)

for x,y,w,h in plate_de:
    area = w*h
    if area > 500:
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow("plate detection",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
