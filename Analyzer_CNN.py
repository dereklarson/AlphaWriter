"""Analyzer_CNN implements a convolutional neural network as a handwriting
classifier.  It requires an external weights file which contains the weights
and biases in an ordered dictionary. 'analyze_writing' is the method that
will take in 'writing' sample and return the character written.
"""


import numpy as np
import cPickle as cp
from collections import OrderedDict as OD
from scipy.ndimage.measurements import center_of_mass as CoM
from scipy.ndimage.filters import correlate as ndCorr
from scipy.misc import imresize as Resize


class Analyzer_CNN(object):

    def __init__(self, param_file):
        self.params = cp.load(open(param_file))
        # Training images are cropped to 28 by 28, so this is our input size
        self.im_size = (28, 28)

    # We crop the data to align with the center of mass and ignore empty space
    # Note: this introduces problems in determining case (e.g. 'c' vs 'C')
    def crop_CoM(self, _in):
        ym, xm = _in.shape
        yc, xc = CoM(_in) # Scipy center of mass

        # Bounding box
        aw = np.argwhere(_in)
        (y0, x0), (y1, x1) = aw.min(0), aw.max(0) + 1

        #A square with dimensions determined by our bounding box, centered
        #at the CoM might stretch over the boundary, so we pad if necessary.
        dim = int(max(abs(y0 - yc), y1 - yc, abs(x0 - xc), x1 - xc))
        padding = int(max(abs(min(xc - dim, yc - dim, 0)),
                    max(xc + dim - xm, yc + dim - ym, 0)))
        _tmp = np.lib.pad(_in, padding, 'constant')

        return _tmp[yc-dim+padding:yc+dim, xc-dim+padding:xc+dim]

    # Feed forward through convolutional neural net
    def predict(self, input):
        _in = input
        for layer in self.params:
            if layer[0] == 'C':
                _w = self.params[layer]['w']
                _b = self.params[layer]['b']
                mid = int(_w.shape[1]/2)
                conv = np.zeros((_w.shape[0],) + _in.shape[1:])
                for i in range(_w.shape[0]):
                    #Convolution here is scipy's correlation op
                    conv[i] = ndCorr(_in, _w[i], mode='constant', cval=0.0)[mid]
                    conv[i] = np.maximum(conv[i] + _b[i], 0)
                _in = np.zeros((_w.shape[0],) + _in.shape[1:])
                _in = max_pool(conv, 2)
            if layer[0] == 'L':
                _in = _in.reshape((-1)) 
                _in = np.dot(_in, self.params[layer]['w']) + self.params[layer]['b']
                _in = np.maximum(_in, 0)
            if layer == 'Out':
                _in = _in.reshape((-1))
                _in = np.dot(_in, self.params[layer]['w']) + self.params[layer]['b']
                return np.argmax(_in)
        return None

    # Method to be called externally
    def analyze_writing(self, matrix):
        cropped = self.crop_CoM(matrix)
        rs = np.array(Resize(cropped, self.im_size), dtype=np.float32)
        rs = rs / 256.
        val = conv_char(self.predict(rs.reshape((1,)+self.im_size)))
        return str(val)

# Converts the NN class designation to a character
def conv_char(val):
    if val < 10:
        return chr(val + ord('0'))
    else:
        val -= 10
        if val % 2 == 0:
            return chr(int(val/2) + ord('a'))
        else:
            return chr(int(val/2) + ord('A'))

# Implements max-pooling, as in CNN: take the max of dx by dx patches
def max_pool(_in, dx):
    assert len(_in.shape) == 3, "Wrong shape for pooling"
    xs, ys = _in.shape[1:]
    assert xs % dx == 0 and ys % dx == 0, "Need commensurate dims for pool"
    ret = np.zeros((_in.shape[0], xs/dx, ys/dx))
    for i in range(0, xs, dx):
        for j in range(0, ys, dx):
            ret[:, i/dx, j/dx] = np.max(_in[:, i:i+dx, j:j+dx].reshape(-1,4), axis=1)
    return ret 

