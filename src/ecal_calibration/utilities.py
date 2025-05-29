'''
Module holding code meant to be reused elsewhere
'''

from importlib.resources import files

from dask                import dataframe
from dask.dataframe      import DataFrame as DDF

import dmu.generic.utilities as gut

# ------------------------------------
def load_cfg(name : str) -> dict:
    '''
    Loads YAML file with configuration and returns dictionary.

    Parameters:

    name (str) : String representing part of the path to config file, e.g. regressor/simple for .../regressor/simple.yaml
    '''

    config_path = files('ecal_calibration_data').joinpath(f'{name}.yaml')
    config_path = str(config_path)
    data        = gut.load_json(config_path)

    return data
# ---------------------------------------------
def _inject_bias(ddf : DDF, bias : float, kind : str) -> DDF:
    '''
    This function scales the momentum components of the lepton by the `bias` factor
    This is done only when the electrons have brem associated, i.e. L*_brem == 1

    Parameters
    --------------
    kind (str) : Type of bias
        flat: bias is uncorrelated with anything
        row : Correlation with the row
    '''
    for lep in ['L1', 'L2']:
        if   kind == 'flat':
            ddf[f'{lep}_PT'] = ddf[f'{lep}_PT'] + ddf[f'{lep}_PT'] * ddf[f'{lep}_brem'] * (bias - 1)
        elif kind == 'row' :
            ddf[f'{lep}_PT'] = ddf[f'{lep}_PT'] + ddf[f'{lep}_PT'] * ddf[f'{lep}_brem'] * (bias - ddf[f'{lep}_BREMHYPOROW'] / 60.)
        else:
            raise ValueError(f'Invalid bias: {kind}')

    return ddf
# ------------------------------------
def get_ddf(bias : float, kind : str) -> DDF:
    '''
    Returns Dask DataFrame with toy data, used for tests

    Parameters
    ---------------
    bias (float): Numerical value of bias, if flat, should be around 1, 1 will be no bias.
    kind (str)  : Type of bias, `flat` for same bias for all electrons, `row` for row dependent one.
    '''
    data_path = files('ecal_calibration_data').joinpath('tests/data/toy_decays.json')
    data_path = str(data_path)
    ddf       = dataframe.read_json(data_path, orient='records', lines=True)
    ddf       = _inject_bias(ddf=ddf, bias=bias, kind=kind)

    return ddf
# ------------------------------------
