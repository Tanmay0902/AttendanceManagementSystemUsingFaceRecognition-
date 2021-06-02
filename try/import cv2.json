import cv2
import face_recognition as fr
imgAng = fr.load_image_file('andrew_ng.jpg')
Test = fr.load_image_file('ian_godfellow.jpg')
fLoc = fr.face_locations(imgAng)[0]
encodeAng = fr.face_encodings(imgAng)[0]
fLocTest = fr.face_locations(Test)[0]
encTest = fr.face_encodings(Test)[0]
result = fr.compare_faces([encodeAng],encTest)
faceDist = fr.face_distance([encodeAng],encTest)
print(result,faceDist)