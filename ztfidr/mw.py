import sncosmo
from extinction import ccm89
import numpy as np


def correct_mwextinction_on_hostmag(hostmag, rv=3.1):
    """ """
    hostmag = hostmag.copy() # do not affect the input original
    av = hostmag["mwebv"] * rv
    bands = ["g","r","i","z","y"]
    wave = np.asarray([sncosmo.get_bandpass(f"ps1::{band}").wave_eff for band in bands])
    offsets = np.asarray([ccm89(wave, av_, 3.1) for av_ in av])
    for i, b_ in enumerate(bands):
        hostmag[f"PS1{b_}"] -= offsets[:,i] # reduces the magnitude as it gets brighter

    return hostmag
