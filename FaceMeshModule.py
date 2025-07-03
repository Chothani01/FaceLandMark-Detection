import cv2
import time
import numpy as np
import mediapipe as mp

class FaceMeshDetector():
    
    def __init__(self, staticMode=False, maxFaces=2, refine_landmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine_landmarks = refine_landmarks
        
#       For refine_landmarks=True,
#       ✅ If you need very detailed eye tracking (e.g., gaze tracking or iris center).
#       ✅ If you want more accurate mouth and eye shape detection.
#       ❌ Slightly slower and uses more computation.

        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refine_landmarks, self.minDetectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpecLm = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(255, 0, 0))
        self.drawSpecConnection = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(255, 0, 255))
        
    
    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLm in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLm, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpecLm, self.drawSpecConnection)
                    # Name What it draws
                    # FACEMESH_TESSELATION	Dense triangular mesh covering whole face.
                    # FACEMESH_CONTOURS	Only main facial contours (jawline, lips, eyes, etc.).
                    # FACEMESH_IRISES	Connections around irises (eye detail).
        return img
    
    
    def findFaceLandmarks(self, img):
        faces_lmList=[]
        if self.results.multi_face_landmarks:
            for faceLm in self.results.multi_face_landmarks:
                face=[]
                for id, lm in enumerate(faceLm.landmark):
                    w, h, c = img.shape
                    x, y = int(lm.x*w), int(lm.y*h)
                    face.append([id, x, y])
                faces_lmList.append(face)
        
        return faces_lmList
    

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        faces = detector.findFaceLandmarks(img)
        if len(faces)!= 0:
            print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF==ord('x'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
