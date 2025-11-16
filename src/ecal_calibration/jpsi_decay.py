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

RANDOM_KEY    : Final        = jax.random.PRNGKey(0)
JPSI_MASS     : Final[float] = 3190.
ELECTRON_MASS : Final[float] = 0.511
PHOTON_MASS   : Final[float] = 0.0

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

        if   self._name in ['e_1', 'e_2']:
            resolution = 0.01
        elif self._name == 'g_1':
            resolution = 0.05
        else:
            raise ValueError(f'Invalid component: {self._name}')

        return value + numpy.random.normal(scale = resolution * abs(value))
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
        pt    = electron.pt * 0.2
        eta   = electron.eta
        phi   = electron.phi

        gamma    = vector.obj(pt=pt, eta=eta, phi=phi, mass=PHOTON_MASS)
        electron = electron - gamma # type: ignore

        return electron, gamma 
    # ----------------------
    def _particles_from_phsp_data(self, data) -> tuple[Momenta, Momenta]: 
        '''
        This is where we transition from the data provided by `phasespace` (messy) to our code

        Parameters
        -------------
        data: Object returned by `generate` method in `phasespace` project when `as_vectors` is True

        Returns
        -------------
        Class storing 4-vectors of electrons
        '''
        try:
            d_particles = data[1]
            e_1 = d_particles['p_0']
            e_2 = d_particles['p_1']
        except Exception as exc:
            log.error(f'Size: {len(data)}')
            log.error(type(data))
            log.error(type(data[1]))
            log.error(data[1].keys())
            raise Exception('Cannot extract dictionary with 4-vectors information from data') from exc

        try:
            l_f_1 : list[tuple[float,float,float,float]] = e_1.tolist()
            l_f_2 : list[tuple[float,float,float,float]] = e_2.tolist()
        except Exception as exc:
            raise Exception('Cannot extract 4-vectors from lists of dictionaries') from exc

        l_e_1 = []
        for px, py, pz, e in l_f_1:
            particle = vector.obj(px=px, py=py, pz=pz, e=e)
            l_e_1.append(particle)

        l_e_2 = []
        for px, py, pz, e in l_f_2:
            particle = vector.obj(px=px, py=py, pz=pz, e=e)
            l_e_2.append(particle)

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
        data['g1_co'] = jax.random.randint(key=RANDOM_KEY, shape=(self._nentries,), minval=0, maxval=10)
        data['g1_rw'] = jax.random.randint(key=RANDOM_KEY, shape=(self._nentries,), minval=0, maxval=10)
        data['g1_ar'] = jax.random.choice( key=RANDOM_KEY, shape=(self._nentries,), a=numpy.array([0, 1, 2]), )
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
