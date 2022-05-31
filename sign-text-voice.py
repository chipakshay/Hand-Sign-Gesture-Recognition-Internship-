# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame
import time
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

print("Hand Gesture Recognition")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

            if className == 'one':
                statement = 'Hello'
                pygame.mixer.init()
                pygame.mixer.music.load('Hello.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()

            elif className == 'two':
                statement = 'Winner'
                pygame.mixer.init()
                pygame.mixer.music.load('Winner.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()

            elif className == 'three':
                statement = 'Water'
                pygame.mixer.init()
                pygame.mixer.music.load('Water.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()

            elif className == 'four':
                statement = 'Four'
                pygame.mixer.init()
                pygame.mixer.music.load('Four.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()
                
            elif className == 'super':
                statement = 'Super'
                pygame.mixer.init()
                pygame.mixer.music.load('Super.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()

            elif className == 'up':
                statement = 'How are you?'
                pygame.mixer.init()
                pygame.mixer.music.load('How are you.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()

            elif className == 'down':
                statement = 'Down'
                pygame.mixer.init()
                pygame.mixer.music.load('Down.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()

            elif className == 'call':
                statement = 'Call'
                pygame.mixer.init()
                pygame.mixer.music.load('Call.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()

            elif className == 'smile':
                statement = 'Smile Please'
                pygame.mixer.init()
                pygame.mixer.music.load('Smile Please.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()
                
            elif className == 'zero':
                statement = 'Stop'
                pygame.mixer.init()
                pygame.mixer.music.load('Stop.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
                pygame.mixer.music.stop()

            else:
                statement = ''
                
            # show the prediction on the frame
            cv2.putText(frame, statement, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
