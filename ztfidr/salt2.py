
# Loads the ztf filters inside sncosmo
import warnings
from ztfquery import filters
filters.load_p48_filters_to_sncosmo(basename="p48") # p48g etc.


from .io import get_salt2params
try:
    SALT2PARAMS = get_salt2params()
except:
    warnings.warn("Failed to load SALT2 parameters")
    SALT2PARAMS = None


    
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
