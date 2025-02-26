import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
detector=cv2.face.LBPHFaceRecognizer.create()
image_paths=[]#enter your own image paths individually
id=[]#assign numerical id with respect to the image path indexes
names={}#make a dictionary assigning id with name eg- 0:"Bob",1:"Alice"
faces=[]
new=[]
for i in range (len(image_paths)):
    img=cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
    face_read=face_cascade.detectMultiScale(img,1.1,5)
    for (x,y,w,h) in face_read:
        faces.append(img[y:y+h,x:x+w])
        print(id[i])
        new.append(id[i])
detector.train(faces,np.array(new))
detector.save('model.yml')
np.save('faces.npy',names)






