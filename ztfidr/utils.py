import numpy as np

def mag_to_flux(mag, magerr=None, units="zp", zp=25.0, wavelength=None):
    """converts magnitude into flux
    Parameters
    ----------
    mag: [float or array]
        AB magnitude(s)
    magerr: [float or array] -optional-
        magnitude error if any
    units: [string] -optional-
        Unit system in which to return the flux:
        - 'zp':   units base on zero point as required for sncosmo fits
        - 'phys': physical units [erg/s/cm^2/A)
    zp: [float or array] -optional-
        zero point of for flux; required if units == 'zp'
    wavelength: [float or array] -optional-
        central wavelength of the photometric filter.
        In Angstrom; required if units == 'phys'
    Returns
    -------
    - float or array (if magerr is None)
    - float or array, float or array (if magerr provided)
    """
    if units not in ["zp", "phys"]:
        raise ValueError("units must be 'zp' or 'phys'")
    elif units == "zp":
        if zp is None:
            raise ValueError("zp must be float or array if units == 'zp'")
        flux = 10 ** (-(mag - zp) / 2.5)
    else:
        if wavelength is None:
            raise ValueError("wavelength must be float or array if units == 'phys'")
        flux = 10 ** (-(mag + 2.406) / 2.5) / wavelength ** 2

    if magerr is None:
        return flux

    dflux = np.abs(flux * (-magerr / 2.5 * np.log(10)))  # df/f = dcount/count
    return flux, dflux
