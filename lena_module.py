import pickle, os

# from array import array
# from pandas import array
from numpy.ma import array


def lena():
    fname = os.path.join(os.path.dirname(__file__), 'lena.dat')
    f = open(fname, 'rb')
    lena = array(pickle.load(f))
    f.close()
    return lena
