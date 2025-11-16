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
    nentries = 100

    obj = JpsiDecay(nentries = nentries)
    df  = obj.get_dataframe()

    columns = set(df.columns.to_list())
    assert columns == {
        'g1_co', 'g1_rw', 'g1_ar',
        'gm_px', 'gm_py', 'gm_pz',
        'e1_px', 'e1_py', 'e1_pz',
        'e2_px', 'e2_py', 'e2_pz',
    } 

    assert len(df) == nentries 
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
