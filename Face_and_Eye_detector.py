import cv2

#img=cv2.imread(r"E:\Downloads from C\multifaces.jfif")

#source=cv2.VideoCapture(r"E:\TV Series\StartUp\@DramaOST.E06.Start-Up.2020.720p.NF.WEB-DL.mkv")
source=cv2.VideoCapture(0)
source.set(cv2.CAP_PROP_POS_MSEC,300000)
face_clsfr=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_clsfr=cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

while(True):
    ret,img=source.read()
    if(ret==False):
        break
    img=cv2.resize(img,(0,0),fx=1,fy=1)
    #blur=cv2.blur(img,(3,3))
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=face_clsfr.detectMultiScale(imgray,1.1,5,minSize=(150,150),maxSize=(350,350))
    #print(faces)
    for face in faces:
        x=face[0]
        y=face[1]
        w=face[2]
        h=face[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,'FACE',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        face_img=imgray[y:y+h,x:x+w]
        eyes=eye_clsfr.detectMultiScale(face_img,1.1,5,minSize=(10,10),maxSize=(100,100))
        for eye in eyes:
            ex = eye[0]
            ey = eye[1]
            ew = eye[2]
            eh = eye[3]
            cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 255), 2)
            cv2.putText(img, 'EYE', (x+ex, y+ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow('IMG',img)
    key=cv2.waitKey(1)
    if (key==27):
        break
cv2.destroyAllWindows()
source.release()
