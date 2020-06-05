import cv2

cas = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# cam = cv2.VideoCapture("name of your video.mp4)
# this will capture the number plates in Real time using your own camera 
cam = cv2.VideoCapture(0)

while True:
    check,image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plate_de = cas.detectMultiScale(gray, 1.1, 5)

    for x, y, w, h in plate_de:
        area = w * h
        if area > 500:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("plate detection", image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
