
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
    
# ================== #
#                    #
#   PARAMETERS       #
#                    #
# ================== #
def get_target_salt2param(targetname):
    """ """
    if SALT2PARAMS is not None:
        if targetname in SALT2PARAMS.index:
            return SALT2PARAMS.loc[targetname]
        else:
            warnings.warn(f"{targetname} is not in the SALTPARAMS dataframe")

    return None

# ================== #
#                    #
#   MODEL            #
#                    #
# ================== #
def get_saltmodel(**params):
    """ """
    import sncosmo
    dust  = sncosmo.CCM89Dust()
    model = sncosmo.Model("salt2", effects=[dust],
                              effect_names=['mw'],
                              effect_frames=['obs'])
    model.set(**params)
    return model
