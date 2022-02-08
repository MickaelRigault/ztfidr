import pandas
import numpy as np
import warnings


from . import io

TARGET_DATA = io.get_targets_data()



class Target():
    """ """
    def __init__(self, lightcurve, spectra, meta=None):
        """ """
        self.set_lightcurve(lightcurve)
        self.set_spectra(spectra)
        self.set_meta(meta)
        
    @classmethod
    def from_name(cls, targetname):
        """ """
        from . import lightcurve, spectroscopy
        lc = lightcurve.LightCurve.from_name(targetname)
        spec = spectroscopy.Spectrum.from_name(targetname, as_spectra=True)
        meta = TARGET_DATA.loc[targetname]
        this = cls(lc, spec, meta=meta)
        this._name = targetname
        return this
        
    
    # -------- #
    #  SETTER  #
    # -------- #    
    def set_lightcurve(self, lightcurve):
        """ """
        self._lightcurve = lightcurve
        
    def set_spectra(self, spectra, run_snid=True):
        """ """
        self._spectra = spectra   
    
    def set_meta(self, meta):
        """ """
        self._meta = meta
        
    # -------- #
    #  GETTER  #
    # -------- #
    def get_snidresult(self, redshift=None, zquality=2, set_it=True, **kwargs):
        """ """
        if redshift is None:
            z_, z_quality_ = self.get_redshift()
            if zquality is not None and \
               zquality not in ["*","all"] and \
               z_quality_ in np.atleast_1d(zquality):
                redshift = z_
            
        phase = self.spectra.get_phase( self.salt2param["t0"] )
        snidres = self.spectra.get_snidfit(phase=phase, redshift=redshift, **kwargs)
        if set_it:
            self.spectra.set_snidresult(snidres)
            
        return snidres
        
    # -------- #
    #  LOADER  #
    # -------- #
    def get_redshift(self, ):
        """ """
        return self.meta["redshift"], self.meta["z_quality"]
    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, spiderkwargs={}):
        """ """
        import matplotlib.pyplot as mpl
        fig = mpl.figure(figsize=[9,6])
        if self.spectra.snidresult is None:
            _ = self.get_snidresult()
            
        # - Axes
        axs = fig.add_axes([0.1,0.6,0.6,0.65/2])
        axt = fig.add_axes([0.75,0.55,0.2,0.75/2], polar=True)
        axlc = fig.add_axes([0.1,0.08,0.85,0.4])

        # - Labels
        phase = self.spectra.get_phase( self.salt2param["t0"] )
        redshift = self.salt2param["redshift"]
        label=rf"{self.name} z={redshift:.3f} | $\Delta$t: {phase:+.1f}"
        
        # - Plotter
        lc = self.lightcurve.show(ax=axlc, 
                                  zprop=dict(ls="-", color="0.6",lw=0.5))
        sp = self.spectra.show_snidresult(axes=[axs, axt], 
                                          label=label, spiderkwargs=spiderkwargs)
        
        # - ObsLine
        axlc.axvline(self.spectra.get_obsdate().datetime, 
                     ls="--", color="0.7")
        return fig
        
    # ================ #
    #    Properties    #
    # ================ #
    @property
    def lightcurve(self):
        """ """
        return self._lightcurve
    
    @property
    def spectra(self):
        """ """
        return self._spectra
    
    @property
    def meta(self):
        """ """
        return self._meta
    
    @property
    def salt2param(self):
        """ shortcut to self.lightcurve.salt2param"""
        return self.lightcurve.salt2param
    
    @property
    def name(self):
        """ """
        if not hasattr(self, "_name"):
            return self.meta.name
        return self._name
