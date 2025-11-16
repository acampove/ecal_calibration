'''
Module with the Objective class
'''
import pandas as pnd
import numpy
import jax

# -------------------------------
class Objective:
    '''
    Class meant to:

    - Pick a pandas dataframe with data representing
        - Two tracks as 4-vectors
        - A photon represented by another 4-vector associated to one of the two tracks

    - Convert dataset into a mass distribution
    - Return proxy for quality of mass distribution as mass scale and resolution 
    '''
    # ----------------------
    def __init__(self, df : pnd.DataFrame) -> None:
        '''
        Parameters
        -------------
        df : Pandas dataframe
        '''
        self._df = df

    # ----------------------
    def _get_masses(self, x : jax.Array) -> numpy.ndarray:
        '''
        Parameters
        -------------
        x: Array with scale factors

        Returns
        -------------
        Array with masses
        '''

        return numpy.random.normal(size=100)
    # ----------------------
    def _get_fom(self, masses : numpy.ndarray) -> float:
        '''
        Parameters
        -------------
        masses: Array with Jpsi masses

        Returns
        -------------
        Figure of merit, used to quantify quality of calibration
        '''
        return masses.std()
    # ----------------------
    def __call__(self, x : jax.Array) -> float:
        '''
        Parameters
        -------------
        x: 1D array symbolizing scale factors, between 0.5 and 1.5

        Returns
        -------------
        Value of loss function, e.g. objective
        '''
        masses = self._get_masses(x=x)
        fom    = self._get_fom(masses=masses)

        return fom 
# -------------------------------
