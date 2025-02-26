import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
detector=cv2.face.LBPHFaceRecognizer.create()
image_paths=["Rishabh\\1.jpg","Rishabh\\2.jpg","Rishabh\\3.jpg","Rishabh\\4.jpg","Rishabh\\5.jpg"
             ,"Rajeev\\1.jpg","Rajeev\\2.jpg","Rajeev\\3.jpg","Rajeev\\4.jpg","Rajeev\\5.jpg"]
id=[0,0,0,0,0,1,1,1,1,1]
names={0:'Rishabh',1:'Rajeev'}
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






