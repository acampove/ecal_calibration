'''
Module holding JpsiDecay class and its dependent class, Momenta
'''
import vector

from dmu.logging.log_store import LogStore
from dmu.generic           import utilities        as gut
from vector                import MomentumObject4D as Momentum
with gut.silent_import():
    import jax
    import phasespace

import numpy
import pandas as pnd

from typing  import Final, Iterator

JPSI_MASS     : Final[float] = 3190.
ELECTRON_MASS : Final[float] = 0.511

log=LogStore.add_logger('ecal_calibration:jpsi_decay')
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
    def _component_from_particle(self, particle : Momentum, component : str) -> float:
        '''
        Parameters
        -------------
        particle: Object representing 4-vector
        component: E.g. px, py, pz, e

        Returns
        -------------
        Numerical value of component
        '''
        value = getattr(particle, component)

        return value + numpy.random.normal(scale = 0.03* abs(value))
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
        if component not in ['px', 'py', 'pz', 'e']:
            raise ValueError(f'Invalid component name: {component}')

        values : list[float] = [ self._component_from_particle(particle, component) for particle in self._particles ]

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
    def _split_electron(self, electron : Momentum) -> tuple[Momentum, Momentum]:
        '''
        Parameters
        -------------
        electron: 4-vector representing electron 

        Returns
        -------------
        Tuple with 4-vectors of electron and emulated brem
        '''

        return electron, electron
    # ----------------------
    def _particles_from_phsp_data(self, data) -> tuple[Momenta, Momenta]: 
        '''
        Parameters
        -------------
        data: Object returned by `generate` method in `phasespace` project when `as_vectors` is True

        Returns
        -------------
        Class storing 4-vectors of electrons
        '''
        try:
            l_vec = [ entry[1] for entry in data ] 
        except IndexError as exc:
            raise IndexError('Cannot extract dictionaries with 4-vectors from data') from exc

        try:
            l_e_1 = [ vec['p0'] for vec in l_vec ]
            l_e_2 = [ vec['p1'] for vec in l_vec ]
        except KeyError as exc:
            raise KeyError('Cannot extract 4-vectors from lists of dictionaries') from exc

        return Momenta(name = 'e_1', particles=l_e_1), Momenta(name = 'e_2', particles=l_e_2)
    # ----------------------
    def get_dataframe(self) -> pnd.DataFrame:
        '''
        Returns
        -------------
        Pandas dataframe with columns representing momenta of particles
        '''
        gen  = phasespace.nbody_decay(mass_top = JPSI_MASS, masses = [ELECTRON_MASS, ELECTRON_MASS])
        data = gen.generate(n_events = self._nentries, as_vectors=True)
        e_1, e_2 = self._particles_from_phsp_data(data=data)

        l_ebrem = []
        l_gamma = []
        for electron in e_1:
            ebrem, gamma = self._split_electron(electron=electron)

            l_ebrem.append(ebrem)
            l_gamma.append(gamma)

        e_1 = Momenta(name='e_1', particles=l_ebrem)
        g_1 = Momenta(name='g_1', particles=l_gamma)

        # --------
        data          = dict()
        data['g1_co'] = jax.random.randint(key=self._key, shape=(self._nentries,), minval=0, maxval=10)
        data['g1_rw'] = jax.random.randint(key=self._key, shape=(self._nentries,), minval=0, maxval=10)
        data['g1_ar'] = jax.random.choice( key=self._key, shape=(self._nentries,), a=numpy.array([0, 1, 2]), )
        # --------
        data['gm_px'] = g_1.as_numpy('px')
        data['gm_py'] = g_1.as_numpy('py')
        data['gm_pz'] = g_1.as_numpy('pz')
        data['gm_e' ] = g_1.as_numpy('e' )
        # --------
        data['e1_px'] = e_1.as_numpy('px')
        data['e1_py'] = e_1.as_numpy('py')
        data['e1_pz'] = e_1.as_numpy('pz')
        data['e1_e' ] = e_1.as_numpy('e' )
        # --------
        data['e2_px'] = e_2.as_numpy('px') 
        data['e2_py'] = e_2.as_numpy('py') 
        data['e2_pz'] = e_2.as_numpy('pz') 
        data['e2_e' ] = e_2.as_numpy('e' )

        df = pnd.DataFrame(data)

        return df
# --------------------------
