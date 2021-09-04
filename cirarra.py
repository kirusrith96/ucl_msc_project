# import standard libraries
import numpy as np # for numeric manipulation 
import pandas as pd # for data manipulation 
from datetime import datetime # for date and time manipulation 
import matplotlib.pyplot as plt # for plotting results 
from matplotlib.ticker import ScalarFormatter
from pylab import * # for plotting results 
import seaborn as sns # for formatting plots 

# check if python library to connect to device is installed
try:
    from rtlsdr import *
except:
    raise ValueError('"pyrtlsdr" library failed to import.')

class trng:
    
    def __init__(self, k=1024, sample_size=2**18, seed=123, path=None, verbose=False):
        """
        A script to generate random numbers (reals in U[0,1]) from atmospheric noise.
        Utilises an RTL-SDR reciever.
        Arguments: 
         - k (the slice level, i.e. if k = 10 we only take every 10th data point and discard the rest)
         - sample_size (the number of IQ samples to take)
         - seed (an integer in the range [1,1e6] that will set the start point of the iterator)
         - path (if None, use live recording; else specify a path to pre-recording IQ values)
        Returns:
         - 
        """
        self.path = path
        if type(seed) == str:
            seed = int(seed)
        self.seed = seed
        
        # get samples from file or generate new ones
        if path==None:
            try:
                # settings
                center_frequency=435e6
                gain=40
                # take a sample
                if verbose == True:
                    print('Sampling at {0} MHz and {1} gain... '.format(int(center_frequency/1e6), gain), end='')
                samples = self._rtl_sampler(10,center_frequency,gain,sample_size,verbose)
                # save the samples down
                date_time = datetime.datetime.now() 
                with open('data/{0}.npy'.format(date_time), 'wb') as file:
                    np.save(file, samples) 
                print('Saved in data/{0}.npy. Please use this as path argument next time.'.format(date_time))
            except:
                raise ValueError('Connection failed. Check device is connected or provide a path to pre-recorded IQ values.')
        else:
            samples = np.load(path)
            
        # apply RBE to the IQ samples
        if verbose == True:
            print('Applying Elias RBE...', end='')
        self.bits = self._elias_rbe(samples)
        if verbose == True:
            print(' {0}% efficiency.'.format(np.round( len(self.bits) / (2*len(samples)) * 100, 2)))
                
        # put the bits into an iterator object
        self.iterator = iter(self.bits)
        # skip ahead bit "seed" items
        for i in range(self.seed):
            next(self.iterator)
     
    # internal method for sampling
    def _rtl_sampler(self, k, f, g, size, verbose=False):
        """
        An internal method to take samples using the RTL-SDR device. Returns samples.
        Arguments: 
         - k (the slice level, i.e. if k = 10 we only take every 10th data point and discard the rest)
         - f (the carrier frequency we wish to sample)
         - g (the level of amplification we are applying to the input signal; default is 40)
         - size (how many samples do we want to take overall; default is 1.048e6)
        Returns:
         - samples (a complex-valued numpy array of IQ data)
        """
        # connect to device
        sdr = RtlSdr()
        # pass arguments
        sdr.center_freq = f
        sdr.gain = g
        sdr.sample_rate=2**20
        sdr.freq_correction=60
        # make a note of the time taken
        if verbose == True:
            start_time = datetime.datetime.now() 
        # repeat until we hit the required sample size
        total_samples = np.asarray([])
        n=0
        fail_counter=10
        while n < size and fail_counter > 0:
            if verbose == True:
                # make a note of what's happening
                now = datetime.datetime.now() 
                duration = now-start_time
                print('Sampling at {0} MHz, {1} gain and k = {2}... '.format(int(f/1e6), g, k), end='')
                print('n = {0} ({1}%). Time elapsed: {2}s'.format(n, np.round((n/size)*100,2), duration.total_seconds()))
            # take samples
            try:
                samples = sdr.read_samples(size)
                # take every kth value
                samples = samples[::k]
                # append samples
                total_samples = np.append(total_samples, samples, axis=0)
                # make a note of what's happening
                n = len(total_samples)
            except:
                fail_counter -= 1
                print('Failed. Retry attempt {}'.format(10-fail_counter))  
        if fail_counter > 0:
            print('Done.')
        else:
            print('Failed. Please check settings and make sure the RTL device is connected.')
    
        # trim samples
        total_samples = total_samples[:size]
        # close connection
        sdr.close()
        return total_samples
    
    # internal method for Elias random bit extraction
    def _elias_rbe(self, samples):
        """
        A random bit extractor that generates unbiased 1's and 0's given a sample of IQ values.
        Arguments: 
         - samples (a complex-valued numpy array of IQ data)
        Returns:
         - bits (a numpy array of binary integers i.e. 1's or 0's)
        """
        el_map = {
            0:[None],
            1:[1,1],
            2:[1,0],
            3:[0,1],
            4:[0,1],
            5:[0],
            6:[0,0],
            7:[0,1],
            8:[0,0],
            9:[1,1],
            10:[1,0],
            11:[1,0],
            12:[1],
            13:[1,1],
            14:[0,0],
            15:[None]
        }
        
        # first turn the IQ data into binary 1's and 0's by simply checking if they are
        # larger than the mean (-> 1) or less than the mean (-> 0)
        x = samples.real
        y = samples.imag
        x_mu = np.mean(x)
        y_mu = np.mean(y)
        xbits = np.heaviside(x, x_mu).astype(int)
        ybits = np.heaviside(y, y_mu).astype(int)
        # turn to strings
        xbits = np.array_split(xbits,len(xbits)/2)
        xbits = [str(x[0]) + str(x[1]) for x in xbits]
        ybits = np.array_split(ybits,len(ybits)/2) 
        ybits = [str(y[0]) + str(y[1]) for y in ybits]
        # join together
        xybits = np.asarray([int(x+y,2) for x,y in zip(xbits,ybits)])
        # apply mapping
        u,inv = np.unique(xybits,return_inverse = True)
        bits = np.array([el_map[x] for x in u], dtype=object)[inv].reshape(xybits.shape)
        # flatten list
        bits = np.concatenate(bits).ravel()
        bits = [x for x in bits if x != None]
        return bits
    
    # internal method for the FDR algorithm
    def _fdr(self, n, flip):
        """
        The Fast Dice Roller algorithm adapted from the pseudocode given in Lumbroso, 2013.
        Arguments: 
         - n (the range of uniform integers we wish to generate)
         - flip (an iterable that returns 1 or 0 when called)
        Returns:
         - c (an integer in Uniform[0,n])
        """
        v, c = 1, 0
        while True:
            v = 2 * v
            c = 2 * c + next(flip)
            if v >= n:
                if c < n:
                    return c
                else:
                    v = v - n
                    c = c - n

   # generate a draw from uniform
    def rand(self, n, verbose=False):
        """
        The main method that returns n draws from uniform[0,1]
        Arguments: 
         - n (the number of draws to be made; must be an integer)
        Returns:
         - u (an array of n uniform[0,1] random variables)
        """
        # use the FDR algorithm to generate uniforms from random bits
        enough_bits = True
        uniforms = []
        if verbose == True:
             print('Applying FDR...', end='')
        while enough_bits and len(uniforms) < n:
            try:
                k = self._fdr(255, self.iterator)
                uniforms.append(k)
            except:
                enough_bits = False
        u = [i / 255 for i in uniforms]
        if verbose == True:
             print(' {0}% efficiency.'.format(np.round( len(u) / (2*len(self.samples)) * 100, 2)))       
        # check if we have enough to give n values back
        if len(u) < n:
            print('Not enough data to generate any more uniform samples. Re-running this method will re-use bits and is not recommended.')
            self.iterator = iter(self.bits)
        else:
            return u[:n]
    
    
    