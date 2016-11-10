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
import scipy as sp
from scipy import signal
import os
# import urllib # python 3
# import urllib2
import pyedflib # http://pyedflib.readthedocs.io/en/latest/
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cbook as cbook

# Machine Learning libraries
#import tensorflow as tf  # https://www.tensorflow.org
#import keras as keras    # https://keras.io


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

def plotEeg(eeg, t, freq, signal_labels, sdata, withSeiz):
    
    # eeg - NxM array - N channels, M samples
    numRows, numSamples = eeg.shape
    # n             = f.signals_in_file
    # signal_labels = f.getSignalLabels()
    # freq          = f.getSampleFrequency(0)
    # numSamples = f.samples_in_file(0)
    # numRows    = n
    
    # t          = np.arange(numSamples, dtype=float)/60.0/freq # this should give minutes
    ticklocs   = []
    ax         = plt.subplot(111)
    plt.xlim(0, numSamples/freq/60.0)
    plt.xticks(np.arange(numSamples/freq/60.0))
    dmin = eeg.min()
    dmax = eeg.max()
    dr   = (dmax - dmin)*0.7  # Crowd them a bit.
    y0   = dmin
    y1   = (numRows - 1) * dr + dmax
    plt.ylim(y0, y1)

    ticklocs = []
    segs = []
    for i in range(numRows):
        segs.append( np.hstack((t[:, np.newaxis], eeg[i, :, np.newaxis])) )
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
    for i in range(len(sdata[withSeiz]['start'])):
        # print i
        plt.axvspan(sdata[withSeiz]['start'][i]/60.0,
                    sdata[withSeiz]['stop' ][i]/60.0, color='red', alpha=0.5)
    plt.show()
    # plt.close()
    
def plotTimeAndFreq(ch, eeg, fs):
    #### BEGIN - Derived from: http://forrestbao.blogspot.com/2009/10/eeg-signal-processing-in-python-and.html
    #ch  = 1 # particualr channel to study 
    eeg = np.array(eeg)
    y   = eeg[:,ch]           # the signal, study channel 'ch'
    L   = len(y)              # signal length
    # fs  = freq               # sampling rate
    T   = 1/fs                # sample time
    t   = np.linspace(1,L,L)*T   # time vector
    
    f = fs*np.linspace(0,L/10,L/10)/L  # single side frequency vector, real frequency up to fs/2
    Y = sp.fft(y)
    
    plt.figure()
    filtered = []
    b        = [] # store filter coefficient
    cutoff   = [0.5,4.0,7.0,12.0,30.0]
    lgndLabels = ['original','delta band, 0-4 Hz','theta band, 4-7 Hz','alpha band, 7-12 Hz','beta band, 12-30 Hz']
    
    for band in xrange(0, len(cutoff)-1):
        wl = 2*cutoff[band]/fs*np.pi
        wh = 2*cutoff[band+1]/fs*np.pi
        M  = 512      # Set number of weights as 128
        bn = np.zeros(M)
     
        for i in xrange(0,M):     # Generate bandpass weighting function
            n = i - M/2       # Make symmetrical
            if n == 0:
                bn[i] = wh/np.pi - wl/np.pi;
            else:
                bn[i] = (np.sin(wh*n))/(np.pi*n) - (np.sin(wl*n))/(np.pi*n)   # Filter impulse response
     
        bn = bn*np.kaiser(M,5.2)  # apply Kaiser window, alpha= 5.2
        b.append(bn)
     
        [w,h] = sp.signal.freqz(bn,1)
        filtered.append(np.convolve(bn, y)) # filter the signal by convolving the signal with filter coefficients
    
    plt.figure(figsize=[16, 10])
    plt.subplot(2, 1, 1)
    plt.plot(y)
    for i in xrange(0, len(filtered)):
        y_p = filtered[i]
        plt.plot(y_p[ M/2:L+M/2])
      
    plt.axis('tight')
    plt.title('Time domain')
    plt.xlabel('Time (seconds)')
    
    plt.subplot(2, 1, 2)
    plt.plot(f,2*abs(Y[0:L/10]))
    for i in xrange(0, len(filtered)):
        Y = filtered[i]
        Y = sp.fft(Y [ M/2:L+M/2])
        plt.plot(f,abs(Y[0:L/10]))
      
    plt.axis('tight')
    plt.legend(lgndLabels)
    
    for i in xrange(0, len(filtered)):   # plot filter's frequency response
        H = abs(sp.fft(b[i], L))
        H = H*1.2*(max(Y)/max(H))
        plt.plot(f, 3*H[0:L/10], 'k')  
      
    plt.axis('tight')
    plt.title('Frequency domain')
    plt.xlabel('Frequency (Hz)')
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.99, top=0.97)
    #pl.tsavefig('filtered.png')
    ##### END - http://forrestbao.blogspot.com/2009/10/eeg-signal-processing-in-python-and.html

def plotTimeAndFreq2(y, Fs, createPlot): #### from: https://plot.ly/matplotlib/fft/
    
    #y = subEeg[0,:]
    #Fs = freq
    #T = seconds
    
    Ts = 1/Fs; # sampling interval
    t = np.arange(0,T,Ts) # time vector
    
    n   = len(y) # length of the signal
    k   = np.arange(n)
    T   = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range
    
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    
    if createPlot:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(t,y)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude')
        ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('|Y(freq)|')
    return [frq,Y]

### from http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
def rolling_window(seq, window_size):
    it = iter(seq)
    win = [it.next() for cnt in xrange(window_size)] # First window
    yield win
    for e in it: # Subsequent windows
        win[:-1] = win[1:]
        win[-1] = e
        yield win
        
###############################################################################
# Begin useful stuff
flag = 1
while flag != -1: flag = pyedflib.close_file(0) # close the file

root            = "/Users/heslote1/www.physionet.org/pn6/chbmit/" # end all folders with "/"
subFolders      = [x[0] for x in os.walk(root)][1:-1]

ok  = 0
bad = 0

seizData = []

for subFolder in subFolders:
    #subf            = "chb01/" # write wrapper
    subf            = os.path.basename(subFolder)
    summaryfile     = os.path.join(subFolder, subf + "-summary.txt")
    sdata, withSeiz = parse_summary(summaryfile)
    for iFile in withSeiz:
        filename        = sdata[iFile]['File Name'] # write wrapper, this just grabs 1st file with seizures
        #filename       = "chb01_03.edf"
        fname           = os.path.join( subFolder, filename)
        # print fname
        flag = 1
        while flag != -1: flag = pyedflib.close_file(0) # close the fi
        try:
            ok += 1
            f               = pyedflib.EdfReader(fname) # os.path.isfile(fname)
        except Exception:
            bad += 1
            print(fname)
            continue
        
         
        # From: http://pyedflib.readthedocs.io/en/latest/
        nSignals      = f.signals_in_file
        nSamples      = f.getNSamples()[0]
        signal_labels = f.getSignalLabels()
        freq          = f.getSampleFrequency(0)
        t             = np.arange(nSamples, dtype=float)/60.0/freq # this should give minutes
        
        # Find and replace all sigbus -> eeg
        eeg       = np.zeros((nSignals, f.getNSamples()[0]))
        sigLbl    = []
        for i in np.arange(nSignals):
            eeg[i, :] = f.readSignal(i)
        
        pyedflib.close_file(0) # close the file to remove errors
            
        #  edited from  http://matplotlib.org/examples/pylab_examples/mri_with_eeg.html
        plt.close('all')
        if 0:   # plot the EEG
            plotEeg(eeg, t, freq, signal_labels, sdata, iFile)
        
        if 0:
            interval = 1
            seconds  = 30.0
            subEeg   = eeg[:,np.array(int(seconds)*range(int(freq))) + (interval-1)]
            plotTimeAndFreq(0, subEeg, freq)
            frq, Y = plotTimeAndFreq2(subEeg[0,:], freq, seconds, 1)
            # plotTimeAndFreq(0, eeg, freq) # not working
    
# Split training and testing
    # grab the seizures
    for i in range(len(sdata[iFile]['start'])):
        eeg[:, sdata[iFile]['start'][i]*freq:sdata[iFile]['stop' ][i]*freq]

#############################
def rolling_window(seq, window_size):
    it = iter(seq)
    win = [it.next() for cnt in xrange(window_size)] # First window
    yield win
    for e in it: # Subsequent windows
        win[:-1] = win[1:]
        win[-1] = e
        yield win

timeseries = np.empty([1,2])
ampseries  = np.empty([1,2])
#seq = t
seconds = 8
window_size = freq*seconds #seconds
#for window  in rolling_window(range(int(eeg.shape[1])), int(window_size)):
for start in range(0, eeg.shape[1], int(window_size))[0:-1]:
    window = [x+start for x in range(int(window_size))]
    for i in range(eeg.shape[0]):
        frq, Y     = plotTimeAndFreq2(eeg[i,window], window_size, freq, 0)
        timeseries = np.vstack((timeseries, np.argsort(abs(Y))[::-1][0:2]))
        ampseries  = np.vstack((ampseries, Y[np.argsort(abs(Y))[::-1][0:2]])) 





    


    

        
