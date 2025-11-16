'''
Script used to test Objective class
'''

import pandas as pnd

from jax                   import numpy     as jnp
from ecal_calibration      import Objective
from ecal_calibration      import JpsiDecay 
from dmu.logging.log_store import LogStore

log=LogStore.add_logger('ecal_calibration:test_objective')
# ----------------------
def test_simple(df : pnd.DataFrame) -> None:
    '''
    Simplest test
    '''
    jdc = JpsiDecay(nentries = 1000)
    df  = jdc.get_dataframe()

    x   = jnp.ones(7000)
    obj = Objective(df=df)
    val = obj(x=x)

    assert val > 0
