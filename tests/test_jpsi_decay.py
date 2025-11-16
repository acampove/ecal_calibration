'''
Module meant to test JpsiDecay class and its dependent, Momenta
'''
import numpy
import vector

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
    nparticles = 2 
    v1  = vector.obj(px=1, py=1, pz=1, e=1)
    v2  = vector.obj(px=2, py=2, pz=2, e=2)

    obj = Momenta(name = 'e_1', particles=[v1, v2])
    px  = obj.as_numpy('px')
    py  = obj.as_numpy('py')
    pz  = obj.as_numpy('pz')

    assert isinstance(px, numpy.ndarray)
    assert isinstance(py, numpy.ndarray)
    assert isinstance(pz, numpy.ndarray)

    assert px.shape == (nparticles,)
    assert py.shape == (nparticles,)
    assert pz.shape == (nparticles,)
