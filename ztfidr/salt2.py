
# Loads the ztf filters inside sncosmo
import warnings
from ztfquery import filters
import numpy as np
import pandas
filters.load_p48_filters_to_sncosmo(basename="p48") # p48g etc.


from .io import get_salt2params
try:
    SALT2PARAMS = get_salt2params()
except:
    warnings.warn("Failed to load SALT2 parameters")
    SALT2PARAMS = None



def salt2result_to_dataframe(result):
    """ """
    fitted = np.in1d(result.param_names, result.vparam_names)
    df = pandas.DataFrame(np.asarray([result.parameters, fitted]).T, 
                          result.param_names, 
                          columns=["values", "fitted"])
    df.fitted = df.fitted.astype(bool)
    
    # - Error
    df = df.merge(pandas.Series(dict(result["errors"]), name="errors"), 
                      left_index=True, right_index=True, how="outer")
    
    # - Cov
    dcov = pandas.DataFrame(result["covariance"], columns=result.vparam_names, index=result.vparam_names)
    dcov.columns ="cov_"+dcov.columns
    
    # - merged
    return df.merge(dcov,  left_index=True, right_index=True, how="outer")
    
