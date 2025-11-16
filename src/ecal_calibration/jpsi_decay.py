'''
Module holding JpsiDecay class and its dependent class, Momenta
'''
from dmu.generic import utilities        as gut
from vector      import MomentumObject4D as Momentum
with gut.silent_import():
    import jax

import numpy
import pandas as pnd
import phasespace

from typing  import Final, Iterator

JPSI_MASS     : Final[float] = 3190.
ELECTRON_MASS : Final[float] = 0.511

# ----------------------
class Momenta:
    '''
    Class meant to represent the momentum of a set of particles
    '''
    # ----------------------
    def __init__(self, name : str, particles : list[Momentum]) -> None:
        '''
        Parameters
        -------------
        name     : Name of particle
        particles: List of 4-vectors associated to particles
        '''
        self._name      = name
        self._particles = particles
    # ----------------------
    def __iter__(self) -> Iterator[Momentum]:
        return iter(self._particles)
    # ----------------------
    def as_numpy(self, component : str) -> numpy.ndarray:
        '''
        Parameters
        -------------
        component: String representing component, i.e. E, px, py, pz

        Returns
        -------------
        Numpy array with numerical value of component
        '''
        if component not in ['px', 'py', 'pz']:
            raise ValueError(f'Invalid component name: {component}')

        values : list[float] = [ getattr(particle, component) for particle in self._particles ]

        return numpy.array(values)
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
    def _split_electrons(self, electrons : Momenta) -> tuple[Momenta, Momenta]:
        '''
        Parameters
        -------------
        electrons: Momenta instance with electrons kinematics 

        Returns
        -------------
        Momenta instance for electron and brem photon
        '''

        return electrons, electrons
    # ----------------------
    def get_dataframe(self) -> pnd.DataFrame:
        '''
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
