import cv2
import time
import mediapipe as mp


cap = cv2.VideoCapture(0)
ptime = 0

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
# By default parameters
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    res, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image=img, landmark_list=faceLm, connections=mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawSpec, connection_drawing_spec=drawSpec)
            # Name What it draws
            # FACEMESH_TESSELATION	Dense triangular mesh covering whole face.
            # FACEMESH_CONTOURS	Only main facial contours (jawline, lips, eyes, etc.).
            # FACEMESH_IRISES	Connections around irises (eye detail).
            for id, lm in enumerate(faceLm.landmark):
                w, h, c = img.shape
                x, y = int(lm.x*w), int(lm.y*h)
                print(id, x, y)
            
    # Frame rate
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    
    cv2.putText(img, text=f"FPS: {int(fps)}", org=(40, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 255, 0), thickness=3)
    cv2.imshow("Face", img)
    
    if cv2.waitKey(1) & 0xFF==ord('x'):
        break