import io
import os
import cv2
from google.cloud import vision_v1p3beta1 as vision
from datetime import datetime

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your json file here'
SOURCE_PATH = 'G:/plates/'

def recognise_license_plate(img_path):
    start_time =datetime.now()
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    img = cv2.resize(img,(800, int((height*800)/width)))
    cv2.imshow('Original image',img)
    cv2.imwrite(SOURCE_PATH+"output.jpg", img)
    img_path= SOURCE_PATH+ "output.jpg"
    client = vision.ImageAnnotatorClient()
    with io.open(img_path, 'rb') as image_file :
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts=response.text_annotations

    for text in texts:
        if len(text.description) == 10:
            license_plate = text.description
            print('License plate ')
            vertices = [(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices]
            cv2.putText(img, license_plate, (200,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            print(vertices)
            cv2.rectangle(img, vertices[0], vertices[2], (0, 255, 0), 3)
            print('Total_time : {}'.format(datetime.now() - start_time))
            cv2.imshow('done ',img)
            cv2.waitKey(0)
    cv2.imwrite(SOURCE_PATH + 'done1.jpg', img)
print('the starting time is as followed')
path = SOURCE_PATH + 'download (2).jpg'
recognise_license_plate(path)
print('end')
