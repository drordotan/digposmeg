#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import division # so that 1/3=0.333 instead of 1/3=0

from psychopy import core, sound

#-- Synchronization sound
SYNC_SOUND_DURATION = .3
SYNC_SOUND_SOA = .6
SYNC_SOUND_NTIMES = 4
syncSound = sound.Sound(value=1000, secs=SYNC_SOUND_DURATION)

#------------------------------------------------------------
# Play a synchronization sound
#
def PlaySyncSound():
    for i in range(SYNC_SOUND_NTIMES):
        syncSound.play()
        core.wait(SYNC_SOUND_SOA)
