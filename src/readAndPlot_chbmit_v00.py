# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:40:52 2016

Remember to open from terminal in Python 2.7

brew install wget
wget -r -np -c -N  -k http://www.physionet.org/pn6/chbmit/

%reset

matlab array -> python numpy array

matlab cell array -> python list

matlab structure -> python dict

@author: heslote1
"""
import numpy as np
# import scipy as sp
# import urllib # python 3
# import urllib2
import pyedflib # http://pyedflib.readthedocs.io/en/latest/
import matplotlib

import matplotlib.pyplot as plt
#from matplotlib.pyplot import *
from matplotlib.collections import LineCollection
import matplotlib.cbook as cbook


#url_root         = "http://www.physionet.org/pn6/chbmit"
#url_withSeizures = url_root + "/RECORDS-WITH-SEIZURES"
#url_all          = url_root + "/RECORDS"
#
#for thisUrl in [url_all, url_withSeizures]:
#    # read in all the names in the file
#    f0 = urllib2.urlopen(thisUrl)
#    # f0 = urllib.request.urlopen(thisUrl) # Python 3
#    f1 = f0.read().splitlines()
#
#    for edfFile in f1:
#        f2 = url_root + "/" + edfFile
#        f  = pyedflib.EdfReader(f2)
#        print(f2)

#if 0:
#    root             = "Users/heslote1/www.physionet.org/pn6/"
#    fWithSeizures = root + "RECORDS-WITH-SEIZURES"
#    fAll          = root + "RECORDS"
#    
#    for thisFile in [fWithSeizures]: #[fAll, fWithSeizures]
#        # read in all the names in the file
#        f0 = open(thisFile, 'r')
#        f1 = f0.read().splitlines()
#    
#        for edfFile in f1:
#            f2 = url_root + "/" + edfFiles
#            f  = pyedflib.EdfReader(f2)
#            print(f2)

def parse_summary( sfile ):
    s        = open(sfile,'r')
    summary  = s.read().splitlines()
    i        = -1
    data     = []
    start    = []
    stop     = []
    found    = 0
    seizures = []
    withSeiz = []
    for line in summary:
        if line == "":
            continue
        if found == 1:
            if line.startswith("File"):
                ftime    = ( dict( [tuple(map(str.strip, line.split(':', 1)))] ) )
                data[i].update( ftime )
            elif line.startswith("Number"):
                numSeiz  = ( dict( [tuple(map(str.strip, line.split(':', 1)))] ) )
                seizures = int(numSeiz['Number of Seizures in File']) 
                if seizures == 0:
                    continue
                withSeiz.append(i)
                data[i].update({'numSeiz': seizures})
            elif line.startswith("Seizure S"):
    #            print(line)
                start.append( float(line.split(' ', -1)[-2]) )
            elif line.startswith("Seizure E"):
                stop.append(  float(line.split(' ', -1)[-2]) )
                seizures -= 1
                if seizures == 0:
                    data[i].update( {'start': start, 'stop': stop} )
                    start = []
                    stop  = []
                    found    =  0
                    seizures = -1
                    continue
        if line.startswith("File Name:"):
            i     += 1
            found  = 1
            fname  = ( dict( [tuple(map(str.strip, line.split(':', 1)))] ) )
            start = []
            stop  = []
            seizures = []
            data.append(fname)
    #        print(line)
    return [data, withSeiz]

###############################################################################
# Begin useful stuff
root            = "/Users/heslote1/www.physionet.org/pn6/chbmit/" # end all folders with "/"
subf            = "chb01/" # write wrapper
summaryfile     = root + subf + subf[0:-1] + "-summary.txt"
sdata, withSeiz = parse_summary(summaryfile)
filename        = sdata[withSeiz[1]]['File Name'] # write wrapper, this just grabs 1st file with seizures
#filename       = "chb01_03.edf"
fname           = root + subf + subf[0:-1] + filename
f               = pyedflib.EdfReader(fname)

# From: http://pyedflib.readthedocs.io/en/latest/
n             = f.signals_in_file
signal_labels = f.getSignalLabels()
freq          = f.getSampleFrequency(0)

sigbufs       = np.zeros((n, f.getNSamples()[0]))
sigLbl        = []
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)
    
#  edited from  http://matplotlib.org/examples/pylab_examples/mri_with_eeg.html
close('all')
if 1:   # plot the EEG
    # load the data
    numSamples = f.samples_in_file(0)
    numRows    = n
    data       = sigbufs
    t          = np.arange(numSamples, dtype=float)/60.0/freq # this should give minutes
    ticklocs   = []
    ax         = plt.subplot(111)
    plt.xlim(0, numSamples/freq/60.0)
    plt.xticks(np.arange(numSamples/freq/60.0))
    dmin = data.min()
    dmax = data.max()
    dr   = (dmax - dmin)*0.7  # Crowd them a bit.
    y0   = dmin
    y1   = (numRows - 1) * dr + dmax
    ylim(y0, y1)

    ticklocs = []
    segs = []
    for i in range(numRows):
        segs.append( np.hstack((t[:, np.newaxis], data[i, :, np.newaxis])) )
        ticklocs.append(i*dr)

    offsets       = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs
    lines         = LineCollection(segs, offsets=offsets,
                           transOffset=None,
                           )

    ax.add_collection(lines)

    # set the yticks to use axes coords on the y axis
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(signal_labels)

    plt.xlabel('time segment (s)')
    
    # Highlight seizures in red
    for i in range(len(sdata[withSeiz[1]]['start'])):
        print i
        plt.axvspan(sdata[withSeiz[1]]['start'][i]/60.0,
                    sdata[withSeiz[1]]['stop' ][i]/60.0, color='red', alpha=0.5)
    plt.show()
    # plt.close()
        