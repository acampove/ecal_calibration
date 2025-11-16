'''
Module meant to test JpsiDecay class and its dependent, Momenta
'''
import numpy
from dmu.generic import utilities as gut
with gut.silent_import():
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
    Check that we can actually extract numpy arrays with
    momenta
    '''
    nparticles= 100
    particles = tf.ones(shape=(nparticles, 4))

    obj = Momenta(particles=particles)
    px  = obj['px']
    py  = obj['py']
    pz  = obj['pz']

    assert isinstance(px, numpy.ndarray)
    assert isinstance(py, numpy.ndarray)
    assert isinstance(pz, numpy.ndarray)

    assert px.shape == (nparticles,)
    assert py.shape == (nparticles,)
    assert pz.shape == (nparticles,)
