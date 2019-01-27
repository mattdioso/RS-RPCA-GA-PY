#!/usr/bin/env python3
import datetime
import os

def outputfunction(options, state, flag):
    dt = datetime.datetime.now().strftme("mmmm-dd-yyyy_HH_MM_SS")
    name = '/Log/' + dt + '_Generation'
    Filename = '%s_%d' % ((os.getcwd() + name), state.Generation)
    optchanged = false
    switcher = {
            'init':'Starting the algorithm',
            'iter':'Iterating ... generation' + state.Generation,
            'interrupt':'Iterating ... generation' + state.Generation,
            'done':'Performing final task'
            }
    print(switcher[flag])

    return state, options, optchanged
