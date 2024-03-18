

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script to predict spiking activity from calcium imaging data

The function "load_neurons_x_time()" loads the input data as a matrix. It can
be modified to load npy-files, mat-files or any other standard format.

The line "spike_prob = cascade.predict( model_name, traces )" performs the
predictions. As input, it uses the loaded calcium recordings ('traces') and
the pretrained model ('model_name'). The output is a matrix with the inferred spike rates.

"""


"""

Import python packages

"""

import os, sys
if 'Demo scripts' in os.getcwd():
    sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
    os.chdir('..')  # change to main directory
print('Current working directory: {}'.format( os.getcwd() ))

from ..cascade2p import checks
checks.check_packages()
from ..cascade2p import cascade # local folder




def cas(traces, model_name):

    print('Number of neurons in dataset:', traces.shape[0])
    print('Number of timepoints in dataset:', traces.shape[1])


    """
    
    Select pretrained model and apply to dF/F data
    
    """

    spike_prob = cascade.predict( model_name, traces )

    return spike_prob



