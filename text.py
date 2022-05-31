#!/usr/bin/env python
import numpy as np
import sys
import cv2
import os
import pygame
import time

from gtts import gTTS
import os

print ("Enter the Text :")
str=input()
##str=input()
print (str)

    
#mtext = 'welcome to india welcome to india welcome to india '
lag = 'en'
myobj = gTTS(text=str, lang=lag, slow =False)
myobj.save("hello.mp3")

pygame.mixer.init()
pygame.mixer.music.load('hello.mp3')
pygame.mixer.music.play()
time.sleep(3)
pygame.mixer.music.stop()
