'''
Module meant to test JpsiDecay class
'''

from rx_calibration import JpsiDecay

# ----------------------
def test_simple() -> None:
    '''
    Simplest test
    '''
    obj = JpsiDecay()
    df  = obj.get_dataframe()
    
