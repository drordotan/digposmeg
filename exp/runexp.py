#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import division # so that 1/3=0.333 instead of 1/3=0
import pandas as pd  # data storage
import sys, pylink, random, codecs, ConfigParser, socket
import csv
import numpy as np

from psychopy import visual, core, data, event, logging, gui, parallel # all needed modules from psychopy
from psychopy.constants import *
from psychopy.info import RunTimeInfo
from psychopy.iohub import Computer

from UnicogIptUtils import *

#=========================================================================
#                   Config file format
#
# The script expects that you put everything in a base directory (defined below
# as BASE_PATH). This should have two sub-directories called "config" and "results".
# Within the "config" directory, each experiment is defined as one INI file.
# Read function loadConfig() to see its format. The INI file also specifies the
# name of the trials file ("trials" parameter in the "data" block), which is
# a CSV file with 3 columns: stimulus, location (0-5), and the expected response.
#=========================================================================

#=========================================================================
#                   Constants
#=========================================================================

if True:  # Can set this to TRUE on the MEG computer, and set the correct PsychoPy version
    psychopyVersion = 1.83
    curr_env = 'fosca'

else:
    PS_INFO = RunTimeInfo()
    verStr = PS_INFO['psychopyVersion'].split('.')
    PS_VER = [int(v) for v in verStr]
    psychopyVersion = PS_VER[0] + float('0.' + verStr[1])

    if socket.gethostname().find('Drors-') >= 0:
        curr_env = 'dror-mac'
    elif socket.gethostname() == 'bilbo':
        curr_env = 'dror-pc'
    else:
        curr_env = 'meg'

trigger_port_type = 'none'
full_screen = False
response_port_left = None
response_port_right = None
response_port = None
response_value_left = None
response_value_right = None

if curr_env == 'dror-mac':
    BASE_PATH = '/data/articles/2-InProgress/DigitPositionMEG'
    LEFT_KEY = 'z'
    RIGHT_KEY = 'slash'
    response_mode = 'keyboard'

elif curr_env == 'dror-pc':
    # Dror's Toshiba
    BASE_PATH = 'C:\\data\\articles\\2-InProgress\\DigitPositionMEG'
    LEFT_KEY = 'z'
    RIGHT_KEY = 'slash'
    response_mode = 'keyboard'

elif curr_env == 'fosca':
    BASE_PATH = '/Users/fosca/Documents/Fosca/Post_doc/Projects/digitpositionMEG/Scripts/analysis/'
    LEFT_KEY = 't'
    RIGHT_KEY = 'y'
    response_mode = 'keyboard'
    trigger_port_type = 'None'
    trigger_port_address = '/dev/tty/ACM0'

elif curr_env == 'meg':
    # MEG computer
    BASE_PATH = 'D:\\manips\\Dror\\data'
    LEFT_KEY = 'w'
    RIGHT_KEY = '!'
    full_screen = True
    trigger_port_type = 'parallel'
    trigger_port_address = 0x0378

    # For responding with two handles
    #response_mode = 'two_handles'
    #response_port_left = 0x0BCE1
    #response_port_right = 0x0379

    # For responding with two buttons of one handle
    response_mode = 'one_handle'
    response_port = 0x0379   # The RIGHT handle
    response_value_left = 1  # ???? TBD
    response_value_right = 2

else:
    print('Unknown environment')
    core.quit()
    sys.exit(1)


IN_PATH = BASE_PATH + os.path.sep + 'config'
OUT_PATH = BASE_PATH + os.path.sep + 'results'

FIXATION_ONSET_REF_STIM_OFFSET = 1
FIXATION_ONSET_REF_RESPONSE = 2

RESPONSE_NONE = 0
RESPONSE_PER_TRIAL = 1
RESPONSE_ASYNC = 2

#-- Configuration parameters (all times are in ms)
class ConfigParams:

    # Stimulus appearance
    stimulusDisplayDuration = 0
    hideStimulusOnResponse = True
    fixationOnsetTime = 0
    fixationOnsetRef = FIXATION_ONSET_REF_RESPONSE
    fixationAlwaysOn = False

    # Visual aspects of the stimulus
    stimHeight = 0
    stimWidth = 0
    stimFont = ''
    stimColor = ''
    stimAlign = []
    stimXPos = []
    nPositions = 0
    oneCharacterPerPosition = True

    # inter-trial interval
    ITI = 0

    # Getting responses
    responses = RESPONSE_NONE   # None, per-trial, or asynchronous
    maxResponseDuration = 0

    # Start-of-block behavior
    playSyncSound = False
    waitAfterSyncSound = 0
    instructions = ''

    # Data source
    trialsFile = None
    trainingTrialsFile = None
    shuffle = False
    response_expr = None

    # Results filename
    outFilenameMask = ''

#=========================================================================
#                   Get configuration
#=========================================================================

#------------------------------------------------------------
# Show user-input dialog
#
def getUserInput():

    while True:
        dlg = gui.Dlg(title="2-digit position experiment")
        if psychopyVersion >= 1.83:
            dlg.addField('Experiment:', choices = ['Select...'] + [f for f in os.listdir(IN_PATH) if f.endswith('.ini')])
        else:
            dlg.addField('Config file:')
        dlg.addField('Subject ID:')
        dlg.addField('Output filename suffix:')
        if psychopyVersion >= 1.83:
            dlg.addField('Save output: ', choices = ['Yes', 'No'])

        dlg.show()
        if not dlg.OK:
            print('User cancelled')
            core.quit()

        configFN = dlg.data[0]
        subjID = dlg.data[1]
        configFnComment = dlg.data[2]
        saveOutput = dlg.data[3] == 'Yes' if psychopyVersion >= 1.83 else True

        if configFN == 'Select...' or subjID == '':
            print('Information was not provided')
        else:
            # OK
            break

    if psychopyVersion < 1.83 and not configFN.endswith('.ini'):
        configFN += ".ini"
    
    return [configFN, subjID, configFnComment, saveOutput]

#------------------------------------------------------------
# Load the experiment configuration
#
def loadConfig(filename):
    print('Loading config file ' + filename)

    # Read config file
    config = ConfigParser.ConfigParser()
    config.readfp(codecs.open(IN_PATH + os.path.sep + filename, "r", "utf8"))
    
    params = ConfigParams()

    params.hideStimulusOnResponse = config.get('behavior', 'hideStimulusOnResponse').lower().startswith('t')
    params.ITI = int(config.get('behavior', 'ITI'))
    params.stimulusDisplayDuration = int(config.get('behavior', 'stimulusDisplayDuration'))
    params.outFilenameMask = config.get('save', 'filename', True)
    params.playSyncSound = config.get('behavior', 'playSyncSound').lower().startswith('t')
    if params.playSyncSound:
        params.waitAfterSyncSound = int(config.get('behavior', 'waitAfterSyncSound'))
    params.shuffle = bool(config.get('data', 'shuffle'))

    getResponses = config.get('behavior', 'user_response').lower()
    if getResponses == 'none':
        params.responses = RESPONSE_NONE
    elif getResponses == 'pertrial':
        params.responses = RESPONSE_PER_TRIAL
    elif getResponses == 'async':
        params.responses = RESPONSE_ASYNC
    else:
        print('Invalid config value [behavior]:user_response - %s' % getResponses)
        core.quit()

    if params.responses == RESPONSE_PER_TRIAL:
        params.maxResponseDuration = int(config.get('behavior', 'maxResponseDuration'))

    params.fixationAlwaysOn = config.get('behavior', 'fixationAlwaysOn').lower().startswith('t') if config.has_option('behavior', 'fixationAlwaysOn') else False
    fixationOnsetRef = config.get('behavior', 'fixationOnsetRef').lower()
    if fixationOnsetRef == 'response':
        # A response immediately offsets the trial and starts fixation of the next trial
        params.fixationOnsetRef = FIXATION_ONSET_REF_RESPONSE
        if params.responses != RESPONSE_PER_TRIAL:
            print('Invalid config: [behavior]/fixationOnsetRef was set to "Response", but per-trial responses are not recorded in this experiment')
            core.quit()
    elif fixationOnsetRef == 'stimoffset':
        params.fixationOnsetRef = FIXATION_ONSET_REF_STIM_OFFSET
    else:
        print('Error: Invalid [behavior]/fixationOnsetRef parameter in config file (%s) - expecting either "Response" or "StimOffset"' % fixationOnsetRef)
        core.quit()
    params.fixationOnsetTime = int(config.get('behavior', 'fixationOnsetTime'))

    # Get instructions
    for i in range(10):
        opt = 'line' + str(i+1)
        if not(config.has_option('instructions', opt)):
            break
        params.instructions = params.instructions + config.get('instructions', opt) + "\n"

    print('Recording user responses: ' + getResponses)
    print('ITI: ' + str(params.ITI))
    print('stimulusDisplayDuration: ' + str(params.stimulusDisplayDuration))
    print('maxResponseDuration: ' + str(params.maxResponseDuration))

    params.trialsFile = config.get('data', 'trials')
    print('Trial data will be loaded from ' + params.trialsFile)

    if config.has_option('data', 'response_expr'):
        params.response_expr = config.get('data', 'response_expr')

    if config.has_option('data', 'training_trials'):
        params.trainingTrialsFile = config.get('data', 'training_trials')
        print('Training trials will be loaded from ' + params.trainingTrialsFile)

    #-- Get stimulus visual definitions
    params.stimHeight = float(config.get('stimulus', 'height'))
    params.stimWidth = float(config.get('stimulus', 'width'))
    params.stimFont = config.get('stimulus', 'font')
    params.stimColor = config.get('stimulus', 'color')
    params.nPositions = int(config.get('stimulus', 'positions'))
    params.oneCharacterPerPosition = config.get('stimulus', 'oneCharacterPerPosition').lower().startswith('t')

    if config.has_option('stimulus', 'align'):
        params.stimAlign = [x for x in config.get('stimulus', 'align').split(',')]
    if config.has_option('stimulus', 'x'):
        params.stimXPos = [int(x) for x in config.get('stimulus', 'x').split(',')]

    return params

#------------------------------------------------------------
# Load the trials file
def loadTrialData(params):

    filePath = IN_PATH + os.path.sep + params.trialsFile
    trialData = loadTrialsFile(filePath, params.responses != RESPONSE_NONE, params.shuffle, False, params.response_expr)

    if params.trainingTrialsFile is not None:
        filePath = IN_PATH + os.path.sep + params.trainingTrialsFile
        trainingData = loadTrialsFile(filePath, params.responses != RESPONSE_NONE, params.shuffle, True)
        trialData = np.append(trainingData, trialData)

    if params.shuffle:
        print('Trials will be presented in random order')
    else:
        print('Trials will be presented in FIXED order')
    
    return trialData

#------------------------------------------------------------
# Load data of trials from a single CSV file
def loadTrialsFile(filePath, loadResponseCol, shuffle, isTraining, response_expr):

    trialData = []

    datafile = open(filePath, 'r')
    reader = csv.DictReader(datafile, delimiter=',')
    hasNumberCol = 'Number' in reader.fieldnames

    for row in reader:
        trial = { 'stimulus' : row['Stimulus'], 'position' : int(row['Position']), 'trigger' : int(row['Trigger']), 'training' : isTraining }
        if loadResponseCol:
            trial['response'] = int(row['ExpectedResponse'])
            if response_expr is not None:
                trial['response'] = eval(response_expr.format(trial['response']))
        if hasNumberCol:
            trial['number'] = int(row['Number'])

        print('Loaded stimulus=[%s] for position %d' % (trial['stimulus'], trial['position']))
        trialData.append(trial)

    datafile.close()

    if shuffle:
        random.shuffle(trialData)

    return trialData


#=========================================================================
#                   Prepare stimulu
#=========================================================================

#------------------------------------------------------------
# Prepare the fixation point stimulus
#
def prepareFixation(win, params):
    return visual.Circle(win, radius=0.05, edges=32, fillColor='Gray', lineColor='Gray', units='deg')


#------------------------------------------------------------
# Prepare the visual fields for the stimulu
#
def prepareStimulusPlaceholders(win, params):

    if params.nPositions > 10:
        print("ERROR: More than 10 stimulus positions are unsupported because the trigger values would be non-unique")
        core.quit()
        sys.exit()

    #-- Default alignment
    if len(params.stimAlign) == 0:
        params.stimAlign = ['center' for i in range(params.nPositions)]

    #-- Default X position
    if len(params.stimXPos) == 0:
        params.stimXPos = [params.stimWidth * (i - params.nPositions/2 + 0.5) for i in range(params.nPositions)]

    placeHolders = []
    for i in range(params.nPositions):
        ph = visual.TextStim(win, text = '', ori = 0,
                             height = params.stimHeight,
                             units = 'deg',
                             pos = (params.stimXPos[i], 0),
                             font = params.stimFont,
                             alignHoriz = params.stimAlign[i],
                             color = params.stimColor)
        placeHolders.append(ph)

    return placeHolders

#=========================================================================
#                   Run experiment
#=========================================================================


#------------------------------------------------------------
# Run the pre-experiment messages
#
def runPreExperiment(win, params, fixation):
    #-- Show preparation messages
    PresentDiagonals(win)
    Message(win, params.instructions)
    win.flip()
    waitForSpaceKey(win)

    Message(win, u"Preparez vous!")
    win.flip()
    waitForSpaceKey(win)

    #-- Wait a little before starting
    startTime = round(core.getTime() * 1000)
    while(round(core.getTime() * 1000) - startTime < params.ITI):
        fixation.draw()
        win.flip()


#------------------------------------------------------------
def showPostExperimentMessage(win):
    PresentDiagonals(win)
    Message(win, u"Merci")
    win.flip()
    waitForSpaceKey(win)


#------------------------------------------------------------
# Run the whole experiment - present the full list of stimuli
#
def presentStimuli(win, params, trials):
    
    results = {}
    responses = {}
    
    trialNum = 0
    
    totalRT = 0
    totalCorrect = 0
    nResponses = 0
    totalExpectedResponses = 0
    saveNumericStimulus = 'number' in trials[0]

    CORRECT_TEXT = ['incorrect', 'correct']
    
    for trial in trials:
        
        trialNum += 1
        expectedResponse = trial['response'] if params.responses != RESPONSE_NONE else -1
        
        #-- Run the trial
        trialResult = runTrial(win, params, trial, trialNum, trialNum == len(trials))
        if trialResult == 'stop':
            trialNum -= 1
            break

        responseTime = trialResult['responseTime']
        stimOnsetTime = trialResult['onset']

        trialTimeInBlock = int(trialResult['onset'] - blockStartTime)
        isCorrect = (trialResult['response'] == expectedResponse) + 0
        totalCorrect += isCorrect

        expectingResponse = trial['response'] != 0
        totalExpectedResponses += expectingResponse

        #-- Store results of this trial
        results.setdefault('target', []).append(trial['stimulus'])
        results.setdefault('training', []).append(1 if trial['training'] else 0)
        results.setdefault('trigger', []).append(trial['trigger'])
        results.setdefault('position', []).append(trial['position'])
        results.setdefault('trialTime', []).append(trialTimeInBlock)
        results.setdefault('duration', []).append(int(trialResult['offset'] - trialResult['onset']))

        if saveNumericStimulus:
            results.setdefault('numeric_target', []).append(trial['number'])

        if (params.responses == RESPONSE_PER_TRIAL):

            rt = -1 if responseTime is None else responseTime - stimOnsetTime
            totalRT += rt
            results.setdefault('response', []).append(trialResult['response'])
            results.setdefault('correct', []).append(isCorrect)
            results.setdefault('rt', []).append(rt)

            print('   Response=%d(%s), RT=%d ms;   Mean: accuracy = %d%%, RT = %d ms' %
                (trialResult['response'], CORRECT_TEXT[isCorrect], int(rt), totalCorrect*100/trialNum, totalRT/trialNum))

        elif params.responses == RESPONSE_ASYNC and trialResult['response'] != 0:
            nResponses += 1
            responseTimeInBlock = round(responseTime - blockStartTime)
            responses.setdefault('time', []).append(responseTimeInBlock)
            print('   Subject responded at %.3f s (%dth time / %d expected)' % (responseTimeInBlock/1000.0, nResponses, totalExpectedResponses))


    if params.responses == RESPONSE_ASYNC:
        print('SUMMARY: Ran %d/%d trials. Subject responded %d times (expected %d)' % (trialNum, len(trials), nResponses, totalExpectedResponses))
    elif params.responses == RESPONSE_PER_TRIAL:
        print('SUMMARY: Ran %d/%d trials. Mean accuracy = %d%%, mean RT = %d ms' % (trialNum, len(trials), totalCorrect * 100 / trialNum, totalRT / trialNum))

    return results, responses


#------------------------------------------------------------
# Run a single trial: show stimulis, get response
#
def runTrial(win, params, trial, trialNum, isLastTrial):
    
    target = trial['stimulus']
    targetPosition = trial['position']
    print("Starting trial #" + str(trialNum) + ": showing [" + target + "] at location " + str(targetPosition))
    
    response = 0 # No response yet

    nPos = prepareStimulus(params, target, targetPosition, trialNum)
    
    showFixation = params.fixationAlwaysOn

    currTime = round(core.getTime() * 1000)
    endOfStimTime = currTime + params.stimulusDisplayDuration
    onsetTime = None
    responseTime = None

    #-- This loop shows the stimulus for the required duration
    while currTime < endOfStimTime:
        
        for i in range(nPos):
            placeHolders[targetPosition+i].draw()
        if showFixation:
            fixation.draw()
        win.flip()
        currTime = round(core.getTime() * 1000)
        if onsetTime is None:
            onsetTime = currTime
        meg_port.setData(trial['trigger'])

        #-- Check if subject responded - but only if need to, and the subject didn't respond previously
        if params.responses != RESPONSE_NONE and response == 0:
            response = responseGetter.checkResponse()
            if response < 0:
                # Clicked ESC to quit
                return 'stop'
            elif (response > 0):
                # Subject made a response
                responseTime = currTime
                if params.hideStimulusOnResponse:
                    break  # Stop presenting the stimulus

        #-- If not checking responses: still check if ESC was clicked
        elif userPressedEscape():
            return 'stop'

    meg_port.setData(0)
    
    
    win.flip() # switch stimulus off
    currTime = round(core.getTime() * 1000)
    offsetTime = currTime

    #-- For deciding when the fixation will be plotted again
    fixationOnsetTime = offsetTime + params.fixationOnsetTime if params.fixationOnsetRef == FIXATION_ONSET_REF_STIM_OFFSET else 0

    #-- Continue waiting for user response.
    #-- We wait for the response for some maximal duration, and then apply inter-trial interval,
    #-- during which the subject is still allowed to respond
    iti = (0 if isLastTrial else params.ITI) # don't apply ITI to the last trial
    endOfTrialTime = currTime + iti
    if response == 0 and params.responses == RESPONSE_PER_TRIAL:
        endOfTrialTime += params.maxResponseDuration # apply additional response time - unless a response was made

    #-- This loop waits for user response
    while currTime < endOfTrialTime:

        if response == 0 and params.responses != RESPONSE_NONE:
            #-- Check if a response was made
            response = responseGetter.checkResponse()
            if response < 0:
                return 'stop'
            
            if (response != 0): # subject responded
                responseTime = currTime
                
                # Once subject - responded don't keep waiting for more than ITI
                endOfTrialTime = min(endOfTrialTime, currTime + iti)

                if params.fixationOnsetRef == FIXATION_ONSET_REF_RESPONSE:
                    # Now that the response was made, start the countdown for showing fixation
                    fixationOnsetTime = currTime + params.fixationOnsetTime
                else:
                    # If fixation was aligned with stimulus offset, present it immediatenly when a response was made
                    showFixation = True

        else:
            #-- Response was already made, or we don't expect any response at all
            #-- Not waiting for response anymore, but keep monitoring ESC presses
            if userPressedEscape():
                return 'stop'

        #-- If fixation did not appear yet: maybe it should appear already
        if not(showFixation) and fixationOnsetTime > 0 and currTime >= fixationOnsetTime:
            showFixation = True

        if showFixation and not isLastTrial:
            fixation.draw()
        
        win.flip()
        currTime = round(core.getTime() * 1000)


    return { 'onset' : onsetTime, 'offset' : offsetTime, 'response' : response, 'responseTime' : responseTime }


#------------------------------------------------------------
# Assign stimulus into placeholders
# Return the number of placeholders used
#
def prepareStimulus(params, target, targetPosition, trialNum):
    
    if targetPosition < 0 or targetPosition >= len(placeHolders):
        print("ERROR in trial #{0}: target position '{1}' is invalid.".format(trialNum, targetPosition))
        core.quit()
    
    #-- Reset placeholders
    for ph in placeHolders:
        ph.setText('')


    if params.oneCharacterPerPosition:
        #-- Different characters are shown in different placeholders
        for i in range(len(target)):
            placeHolders[targetPosition+i].setText(target[i])
        return len(target)

    else:
        #-- The whole stimulus is in a single placeholder
        placeHolders[targetPosition].setText(target)
        return 1


#------------------------------------------------------------
# Connect to the MEG's port, to allow sending triggers
#
def createMEGPort():
    if trigger_port_type == 'parallel':
        try:
            port = parallel.ParallelPort(address=trigger_port_address)
            print('Connected to the MEG trigger port, resetting...')
            port.setData(0)
            print('MEG trigger port is ready')
            return port
        except:
            print('Problem connecting to the MEG trigger port')
            print sys.exc_info()
            core.quit()

    elif trigger_port_type == 'serial':
        try:
            port = TriggerSerialPort(trigger_port_address)
            print('Connected to the MEG trigger port, resetting...')
            port.setData(0) # fix this!!!
            print('MEG trigger port is ready')
            return port
        except:
            print('Problem connecting to the MEG trigger port')
            print sys.exc_info()
            core.quit()

    else:
        return MockPort()

#------------------------------------------------------------
#-- Save results to CSV file
#
def saveResults(results, outFN):
    results = pd.DataFrame(results)
    results.to_csv(outFN, index=False, encoding = 'latin-1')
    print("Results were saved to " + outFN)

#=========================================================================
#              Main program
#=========================================================================

# Set the python process to high priority
Computer.enableHighPriority()

if not os.path.isdir(BASE_PATH):
    print('Error: the path is wrong (%s). Either you changed the app path or I don''t know this computer' % BASE_PATH)
    core.quit()

#-- Ask user what to run
[configFN, subjID, configFnComment, saveOutput] = getUserInput()

print("Starting...")

#-- Setup
monitorSize = (1024, 768)
win = visual.Window(size = monitorSize, fullscr = full_screen, monitor="testMonitor", units="pix", color='Black', screen = 1)
myMouse = event.Mouse(win=win)
myMouse.setVisible(0) # Make mouse invisible

#-- Load configuration
print("Load config...")
params = loadConfig(configFN)
trialData = loadTrialData(params)

outFN = params.outFilenameMask % { 'subjid' : subjID, 'comment' : configFnComment, 'time' :  core.getAbsTime() }
outFN = OUT_PATH + os.path.sep + outFN
print('Results will be saved to %s' % outFN)

if params.responses == RESPONSE_ASYNC:
    responsesOutFN = params.outFilenameMask % { 'subjid' : subjID + '-responses', 'comment' : configFnComment, 'time' :  core.getAbsTime() }
    responsesOutFN = OUT_PATH + os.path.sep + responsesOutFN

#-- Prepare visual elements
print("Prepare visual elements...")
fixation = prepareFixation(win, params)
placeHolders = prepareStimulusPlaceholders(win, params)

#-- Initialize I/O
print("Initialize I/O...")
meg_port = createMEGPort()
if (params.playSyncSound):
    from UnicogSoundUtils import *


if response_mode == 'keyboard' or params.responses == RESPONSE_NONE:
    responseGetter = ResponseFromKeyboard(LEFT_KEY, RIGHT_KEY)
elif response_mode == 'two_handles':
    responseGetter = ResponseFromTwoPorts(response_port_left, response_port_right)
elif response_mode == 'one_handle':
    responseGetter = ResponseFromOnePort(response_port, response_value_left, response_value_right)
else:
    raise Exception('Invalid responseMode ({:})'.format(response_mode))


print("Prepare subject")

#-- Message to subject
runPreExperiment(win, params, fixation)


#========================
#== Run the experiment ==
#========================

print("Starting experiment block!")

blockStartTime = round(core.getTime() * 1000)

if (params.playSyncSound):
    PlaySyncSound()

results, responses = presentStimuli(win, params, trialData)

print("Block ended after " + str(round(core.getTime() - blockStartTime/1000)) + " seconds")

#=======================
#== End of experiment ==
#=======================


#-- Save results
if saveOutput:
    saveResults(results, outFN)
    if params.responses == RESPONSE_ASYNC:
        saveResults(responses, responsesOutFN)

#-- Message to subject
showPostExperimentMessage(win)

print("-- Finished running experiment block %s" % configFN)

core.quit()
