#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import division # so that 1/3=0.333 instead of 1/3=0
import os # for file/folder operations
import time
import serial


from psychopy import visual, core, data, event, logging, gui, parallel
from psychopy.constants import *   

#-- Stimuli
textColor = 'White'
textFont = 'Verdana'


#-- Diagonal lines fixation
lineFixationColor = 'Gray'
lineFixationThick = 0.8

#--------------------------------------------------------------------------
def PresentDiagonals(win):
    winSize = win.size
    w = winSize[0]
    h = winSize[1]
    line1 = visual.Line(win, start=(-w/2, -h/2), 
                            end=(w/2, h/2),
                            lineColor = lineFixationColor,
                            lineWidth=lineFixationThick,
                            units='deg')
    line2 = visual.Line(win, start=(-w/2, h/2), 
                            end=(w/2, -h/2),
                            lineColor = lineFixationColor,
                            lineWidth=lineFixationThick,
                            units='deg')
    line1.draw()
    line2.draw()


#--------------------------------------------------------------------------
# Present a text message
def Message(win, string):
    message = visual.TextStim(win,  text=string,
                                    height=0.6,    
                                    ori=0,
                                    pos=(0, 0),
                                    font=textFont,
                                    color=textColor,
                                    units='deg')
    message.draw()
    return message


# ------------------------------------------------------------
# Dummy port
class MockPort(object):
    def __init__(self):
        pass

    def readData(self):
        return 0

    def setData(self, data):
        return


# ------------------------------------------------------------
# Serial port wrapper
class TriggerSerialPort(object):

    def __init__(self, portid):
        self._port = serial.Serial(portid)

    def setData(self, data):
        self._port.write(data)
        self._port.flush()


#------------------------------------------------------------
# Get response from response buttons (parallel port)
#
class ResponseFromTwoPorts(object):
    
    port1 = []
    port2 = []
    
    #--------------------
    #-- Constructor
    def __init__(self, port1Num, port2Num):
        try:
            self.port1 = parallel.ParallelPort(address=port1Num)
            self.port2 = parallel.ParallelPort(address=port2Num)
        except:
            print('Problem connecting to parallel port')
            self.port1 = MockPort()
            self.port2 = MockPort()

    #----------------------------------------
    # Check if subject responded.
    # Return 0 if not; 1 or 2 if they did; and -1 if they clicked ESC
    def checkResponse(self):
        if userPressedEscape():
            return -1
        
        #-- Check if exactly one button was pressed
        resp1 = self.port1.readData()
        resp2 = self.port2.readData()
        if (resp1 != 0 and resp2 == 0):
            return 1
        elif (resp1 == 0 and resp2 != 0):
            return 2
        else:
            return 0


#------------------------------------------------------------
# Get response from two response buttons of the same response handle (same parallel port)
#
class ResponseFromOnePort(object):

    port1 = []
    port2 = []

    #--------------------
    #-- Constructor
    def __init__(self, portNum, resp1Value, resp2Value):
        self.resp1Value = resp1Value
        self.resp2Value = resp2Value
        try:
            self.port = parallel.ParallelPort(address=portNum)
        except:
            print('Problem connecting to parallel port')
            self.port = MockPort()

    #----------------------------------------
    # Check if subject responded.
    # Return 0 if not; 1 or 2 if they did; and -1 if they clicked ESC
    def checkResponse(self):
        if userPressedEscape():
            return -1

        #-- Check if exactly one button was pressed
        resp = self.port.readData()
        if resp == self.resp1Value:
            return 1
        elif resp == self.resp2Value:
            return 2
        else:
            if resp != 0:
                print("Unknown button was clicked: {:}".format(resp))
            return 0


#------------------------------------------------------------
# Get response from keyboard
#
class ResponseFromKeyboard:

    key1 = ''
    key2 = ''

    #---------------
    def __init__(self, k1, k2):
        self.key1 = k1.lower()
        self.key2 = k2.lower()

    #----------------------------------------
    # Check if subject responded.
    # Return 0 if not; 1 or 2 if they did; and -1 if they clicked ESC
    def checkResponse(self):
        
        keys = [x.lower() for x in event.getKeys()]
        
        if len(keys) == 0:
            return 0
        
        resp1 = self.key1 in keys
        resp2 = self.key2 in keys

        if 'escape' in keys:
            print("User pressed ESC, quitting experiment")
            return -1
        if resp1 and not resp2:
            return 1
        elif resp2 and not resp1:
            return 2
        else:
            return 0

#------------------------------------------------------------
#-- Wait until user clicks spacebar [escape will close the experiment]
#
def waitForSpaceKey(win):
    event.clearEvents()
    while True:
        keys = event.getKeys()
        if 'space' in keys:
            return
        if 'escape' in keys:
            print("User pressed ESC, quitting experiment")
            win.close()
            core.quit()
        
        time.sleep(0.2)

#------------------------------------------------------------
#-- If user clicked escape, close the experiment
#
def userPressedEscape():
    stopNow = 'escape' in event.getKeys()
    if stopNow:
        print("User pressed ESC, quitting experiment")
    return stopNow
