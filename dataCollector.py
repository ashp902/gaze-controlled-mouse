from calendar import c
import cv2
import numpy as np
import mediapipe as mp
import time
import csv
import random
from multiprocessing import Process
import threading

def getRightEye(faceLms) :
    return mp.framework.formats.landmark_pb2.NormalizedLandmarkList(
        landmark = [
            faceLms.landmark[33],
            faceLms.landmark[7],
            faceLms.landmark[163],
            faceLms.landmark[144],
            faceLms.landmark[145],
            faceLms.landmark[153],
            faceLms.landmark[154],
            faceLms.landmark[155],
            faceLms.landmark[133],
            faceLms.landmark[173],
            faceLms.landmark[157],
            faceLms.landmark[158],
            faceLms.landmark[159],
            faceLms.landmark[160],
            faceLms.landmark[161],
            faceLms.landmark[246],
        ]
    )

def getLeftEye(faceLms) :
    return mp.framework.formats.landmark_pb2.NormalizedLandmarkList(
        landmark = [
            faceLms.landmark[263],
            faceLms.landmark[249],
            faceLms.landmark[390],
            faceLms.landmark[373],
            faceLms.landmark[374],
            faceLms.landmark[380],
            faceLms.landmark[381],
            faceLms.landmark[382],
            faceLms.landmark[362],
            faceLms.landmark[398],
            faceLms.landmark[384],
            faceLms.landmark[385],
            faceLms.landmark[386],
            faceLms.landmark[387],
            faceLms.landmark[388],
            faceLms.landmark[466],
        ]
    )

def getFace(faceLms) :
    return mp.framework.formats.landmark_pb2.NormalizedLandmarkList(
        landmark = [
            faceLms.landmark[10],
            faceLms.landmark[338],
            faceLms.landmark[297],
            faceLms.landmark[332],
            faceLms.landmark[284],
            faceLms.landmark[251],
            faceLms.landmark[389],
            faceLms.landmark[264],
            faceLms.landmark[447],
            faceLms.landmark[366],
            faceLms.landmark[401],
            faceLms.landmark[435],
            faceLms.landmark[367],
            faceLms.landmark[397],
            faceLms.landmark[365],
            faceLms.landmark[379],
            faceLms.landmark[378],
            faceLms.landmark[400],
            faceLms.landmark[377],
            faceLms.landmark[152],
            faceLms.landmark[148],
            faceLms.landmark[176],
            faceLms.landmark[149],
            faceLms.landmark[150],
            faceLms.landmark[136],
            faceLms.landmark[172],
            faceLms.landmark[215],
            faceLms.landmark[177],
            faceLms.landmark[137],
            faceLms.landmark[227],
            faceLms.landmark[127],
            faceLms.landmark[162],
            faceLms.landmark[21],
            faceLms.landmark[54],
            faceLms.landmark[103],
            faceLms.landmark[67],
            faceLms.landmark[109],
        ]
    )

def getRightIris(faceLms) :
    return mp.framework.formats.landmark_pb2.NormalizedLandmarkList(
        landmark = [
            faceLms.landmark[469],
            faceLms.landmark[470],
            faceLms.landmark[471],
            faceLms.landmark[472],
        ]
    )

def getLeftIris(faceLms) :
    return mp.framework.formats.landmark_pb2.NormalizedLandmarkList(
        landmark = [
            faceLms.landmark[474],
            faceLms.landmark[475],
            faceLms.landmark[476],
            faceLms.landmark[477],
        ]
    )

def getFrame(frame, lms) :
    xmin = int(min([lm.x for lm in lms.landmark]) * frame.shape[1])
    xmax = int(max([lm.x for lm in lms.landmark]) * frame.shape[1])
    ymin = int(min([lm.y for lm in lms.landmark]) * frame.shape[0])
    ymax = int(max([lm.y for lm in lms.landmark]) * frame.shape[0])
    return frame[ymin:ymax, xmin:xmax]

def lmsToList(frame, lms) :
    return [[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in lms.landmark]

px, py = 0.5, 0.5

class RandGen(threading.Thread) :
    def run(self) :
        for _ in range(10) :
            px, py = random.random(), random.random()
            print(px, py)
            time.sleep(3)
        px, py = 0.5, 0.5

vid = cv2.VideoCapture('demo.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(refine_landmarks = True)

data = open('data.csv', 'a')
writer = csv.writer(data)

class Vision(threading.Thread) :
    def run(self) :
        while True:
            _, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.circle(frame, (int(frame.shape[1] * px), int(frame.shape[0] * py)), 100, (255, 0, 0), thickness=50)
            print(px, py)
            results = faceMesh.process(frame)
            if results.multi_face_landmarks :
                for faceLms in results.multi_face_landmarks :
                    '''faceOutlineLms = getFace(faceLms)
                    rightEyeLms = getRightEye(faceLms)
                    rightIris = getRightIris(faceLms)
                    leftEyeLms = getLeftEye(faceLms)
                    leftIris = getLeftIris(faceLms)'''

                    faceList = np.array(lmsToList(frame, getFace(faceLms)), np.int32)
                    rightEyeList = np.array(lmsToList(frame, getRightEye(faceLms)), np.int32)
                    leftEyeList = np.array(lmsToList(frame, getLeftEye(faceLms)), np.int32)
                    rightIrisList = np.array(lmsToList(frame, getRightIris(faceLms)), np.int32)
                    leftIrisList = np.array(lmsToList(frame, getLeftIris(faceLms)), np.int32)

                    #cv2.polylines(frame, [faceList], True, (255, 0, 0), 2)
                    #cv2.polylines(frame, [rightEyeList], True, (255, 0, 0), 2)
                    #cv2.polylines(frame, [leftEyeList], True, (255, 0, 0), 2)
                    #cv2.polylines(frame, [rightIrisList], True, (255, 0, 0), 2)
                    #cv2.polylines(frame, [leftIrisList], True, (255, 0, 0), 2)
                    
                    vector = np.concatenate((faceList, rightEyeList, leftEyeList, rightIrisList, leftIrisList), axis = None)
                    writer.writerow(vector)
                    #print(len(vector))
                    #print('---')
                    #break

                    '''rightEyeFrame = getFrame(frame, rightEyeLms)
                    rightEyeFrame = cv2.cvtColor(rightEyeFrame, cv2.COLOR_RGB2GRAY)

                    rightEyeMask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                    cv2.polylines(rightEyeMask, [rightEyeList], True, (255, 255, 255), 2)
                    cv2.fillPoly(rightEyeMask, [rightEyeList], (255, 255, 255))
                    rightEyeMask = getFrame(rightEyeMask, rightEyeLms)

                    rightEyeFrame = cv2.bitwise_and(rightEyeFrame, rightEyeMask)
                    rightEyeFrame = cv2.GaussianBlur(rightEyeFrame, (7, 7), 0)
                    cv2.imshow('rightEye', rightEyeFrame)
                    _, threshold = cv2.threshold(rightEyeFrame, 30, 255, cv2.THRESH_BINARY)
                    cv2.imshow('thres', threshold)

                    #gray_rightEyeFrame = cv2.cvtColor(rightEyeFrame, cv2.COLOR_RGB2GRAY)
                    #gray_rightEyeFrame = cv2.GaussianBlur(gray_rightEyeFrame, (7, 7), 0)
                    #_, threshold = cv2.threshold(gray_rightEyeFrame, 30, 255, cv2.THRESH_BINARY_INV)
                    #threshold = cv2.resize(threshold, None, fx=5, fy=5)
                    #cv2.imshow('rightEye', gray_rightEyeFrame)
                    #cv2.imshow('thres', threshold)
                    #cv2.imshow('mask', rightEyeMask)'''

            '''cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime'''

            frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
            #frame = cv2.resize(frame, None, fx=5, fy=5)

            #cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.imshow('frame', frame)
            #break
            k = cv2.waitKey(30) & 0xff
            if k == 113 :
                break

        vid.release()
        cv2.destroyAllWindows()

a = RandGen()
b = Vision()

a.start()
b.start()

a.join()
b.join()




















'''face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

vid = cv2.VideoCapture(0)

while True :
    ret, frame = vid.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame)
    eyes = eye_cascade.detectMultiScale(gray_frame)
    if len(faces) != 0 :
        (fx, fy, fw, fh) = faces[0]
        cv2.imshow('face', frame[fy:fy+fh, fx:fx+fw])
    if len(eyes) != 0 :
        (ex, ey, ew, eh) = eyes[0]
        eye = frame[ey:ey+eh, ex:ex+ew]
        eye = cv2.resize(eye, None, fx = 5, fy = 5)
        cv2.imshow('eye', eye)

    #cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 113 :
        break
vid.release()
cv2.destroyAllWindows()'''