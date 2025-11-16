'''
Module with the Objective class
'''
import pandas as pnd
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
    def __call__(self, x : jax.Array) -> float:
        '''
        Parameters
        -------------
        x: 1D array symbolizing scale factors, between 0.5 and 1.5

        Returns
        -------------
        Value of loss function, e.g. objective
        '''
        return 1.0 
# -------------------------------
