'''
Module meant to test JpsiDecay class and its dependent, Momenta
'''
import tensorflow as tf

from ecal_calibration import JpsiDecay
from ecal_calibration import Momenta 

# ----------------------
def test_simple_jpsi_decay() -> None:
    '''
    Simplest test
    '''
    obj = JpsiDecay()
    df  = obj.get_dataframe()
# ----------------------
def test_simple_momenta() -> None:
    '''
    Simplest test for Momenta class
    '''
    particles = tf.ones(shape=(2, 4))

    obj = Momenta(particles=particles)
    px  = obj['px']
    py  = obj['py']
    pz  = obj['pz']
