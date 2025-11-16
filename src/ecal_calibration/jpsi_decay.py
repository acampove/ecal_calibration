'''
Module holding JpsiDecay class and its dependent class, Momenta
'''
from dmu.generic import utilities as gut
with gut.silent_import():
    from tensorflow  import Tensor

import jax
import numpy
import pandas as pnd
import phasespace

from dataclasses import dataclass
#from jax         import numpy as jnp
from typing      import Final

JPSI_MASS     : Final[float] = 3190.
ELECTRON_MASS : Final[float] = 0.511

# ----------------------
@dataclass
class Momenta:
    '''
    Class meant to represent the momentum of a set of particles
    '''
    particles : Tensor # Here the vectors with the momenta of each particle will be stored
    # ----------------------
    def __getitem__(self, key : str) -> numpy.ndarray:
        '''
        Parameters
        -------------
        key: String representing momentum component, e.g. px, py, pz 

        Returns
        -------------
        Numpy array with numerical values of momentum component
        '''
        if key not in ['px', 'py', 'pz']:
            raise ValueError(f'Invalid momentum component: {key}')

        index = {'px' : 1, 'py' : 2, 'pz' : 3}[key]
        matrix= self.particles.numpy()

        return matrix[:, index]
# --------------------------
class JpsiDecay:
    '''
    Class meant to provide datasets symbolizing decays
    of Jpsi mesons into electrons and optionally Bremsstrahlung emulation
    '''
    # ----------------------
    def __init__(self, nentries : int) -> None:
        '''
        Parameters
        -------------
        nentries: Number of entries
        '''
        self._key      = jax.random.PRNGKey(0)
        self._nentries = nentries 
    # ----------------------
    def _split_electrons(self, electrons : Tensor) -> tuple[Momenta, Momenta]:
        '''
        Parameters
        -------------
        

        Returns
        -------------
        
        '''
        
    # ----------------------
    def get_dataframe(self) -> pnd.DataFrame:
        '''
        Parameters
        -------------
        

        Returns
        -------------
        Pandas dataframe with columns representing momenta of particles
        '''

        gen           = phasespace.nbody_decay(mass_top = JPSI_MASS, masses = [ELECTRON_MASS, ELECTRON_MASS])
        _, particles  = gen.generate(n_events = self._nentries)

        e_1           = Momenta(particles = particles['p_0'])
        e_2           = Momenta(particles = particles['p_1'])
        e_1, gamma    = self._split_electrons(electrons = e_1)

        # --------
        data          = dict()
        data['g1_co'] = jax.random.randint(key=self._key, shape=(self._nentries,), minval=0, maxval=10)
        data['g1_rw'] = jax.random.randint(key=self._key, shape=(self._nentries,), minval=0, maxval=10)
        data['g1_ar'] = jax.random.choice( key=self._key, shape=(self._nentries,), a=numpy.array([0, 1, 2]), )
        # --------
        data['gm_px'] = gamma['px']
        data['gm_py'] = gamma['py']
        data['gm_pz'] = gamma['pz']
        # --------
        data['e1_px'] =   e_1['px']
        data['e1_py'] =   e_1['py']
        data['e1_pz'] =   e_1['pz']
        # --------
        data['e2_px'] =   e_2['px'] 
        data['e2_py'] =   e_2['py'] 
        data['e2_pz'] =   e_2['pz'] 

        df = pnd.DataFrame(data)

        return df
# --------------------------
