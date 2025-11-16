'''
Module meant to test JpsiDecay class and its dependent, Momenta
'''
import numpy
import vector
import pandas            as pnd
import matplotlib.pyplot as plt

from vector           import MomentumObject4D as Momentum
from ecal_calibration import JpsiDecay
from ecal_calibration import Momenta 

# ----------------------
def _vector_from_df(name : str, df : pnd.DataFrame) -> list[Momentum]:
    '''
    Parameters
    -------------
    name: Particle name, needed to pick column in dataframe
    df  : Dataframe with kinematics

    Returns
    -------------
    List of 4-vectors
    '''
    arr_px = df[f'{name}_px'].to_numpy()
    arr_py = df[f'{name}_py'].to_numpy()
    arr_pz = df[f'{name}_pz'].to_numpy()
    arr_en = df[f'{name}_e' ].to_numpy()

    l_particle = []
    for px, py, pz, en in zip(arr_px, arr_py, arr_pz, arr_en):
        particle = vector.obj(px=px, py=py, pz=pz, e=en)
        l_particle.append(particle)

    return l_particle
# ----------------------
def _plot_mass(df : pnd.DataFrame) -> None:
    '''
    Parameters
    -------------
    df : Pandas dataframe with kinematics
    '''
    l_e1 = _vector_from_df(name='e1', df=df)
    l_e2 = _vector_from_df(name='e2', df=df)

    l_mass = []
    for e1, e2 in zip(l_e1, l_e2):
        jpsi : Momentum = e1 + e2 # type: ignore
        l_mass.append(jpsi.m)

    plt.hist(l_mass, bins=100)
    plt.show()
# ----------------------
def test_simple_jpsi_decay() -> None:
    '''
    Simplest test
    '''
    nentries = 10000

    obj = JpsiDecay(nentries = nentries)
    df  = obj.get_dataframe()

    columns = set(df.columns.to_list())
    assert columns == {
        'g1_co', 'g1_rw', 'g1_ar',
        'gm_px', 'gm_py', 'gm_pz', 'gm_e',
        'e1_px', 'e1_py', 'e1_pz', 'e1_e',
        'e2_px', 'e2_py', 'e2_pz', 'e2_e',
    } 

    assert len(df) == nentries 

    _plot_mass(df=df)
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
