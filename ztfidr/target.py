import pandas
import numpy as np
import warnings


from . import io

TARGET_DATA = io.get_targets_data()

COLORS = ["0.7", "tan", "lightsteelblue", "thistle", "darkseagreen"]

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
        spec = spectroscopy.Spectrum.from_name(targetname, as_spectra=False)
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
        if not self.has_spectra():
            warnings.warn("No spectra loaded.")
            return None
        
        if redshift is None:
            z_, z_quality_ = self.get_redshift()
            if zquality is not None and \
               zquality not in ["*","all"] and \
               z_quality_ in np.atleast_1d(zquality):
                redshift = z_
            
        snidres = []
        for spec_ in self.spectra:
            phase = spec_.get_phase( self.salt2param["t0"] )
            snidres_ = spec_.get_snidfit(phase=phase, redshift=redshift, **kwargs)
            if set_it:
                spec_.set_snidresult(snidres_)
            snidres.append(snidres_)
                
        return snidres[0] if not self.has_multiple_spectra() else snidres
        
    # -------- #
    #  LOADER  #
    # -------- #
    def get_redshift(self, ):
        """ """
        return self.meta["redshift"], self.meta["z_quality"]
    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, spiderkwargs={}, nbest=3):
        """ """
        import matplotlib.pyplot as mpl
        n_speclines = np.max([1,self.nspectra])
        fig = mpl.figure(figsize=[9,3+2.5*n_speclines])

        # - Axes
        _lc_height = 0.4/np.sqrt(n_speclines)
        _sp_height = 0.25/np.sqrt(n_speclines)
        _spany = 0.04+0.08/np.sqrt(n_speclines)
        _lc_spany = 0.02+0.08/np.sqrt(n_speclines)
        _bottom_lc = 0.04+0.05/np.sqrt(n_speclines)
        _top_lc = _bottom_lc+_lc_height
        
        axlc = fig.add_axes([0.1, _bottom_lc, 0.85, _lc_height])
        axes = []
        for i in range(n_speclines):
            axs = fig.add_axes([0.10, _top_lc+_lc_spany+i*(_spany+_sp_height), 0.6, _sp_height])
            axt = fig.add_axes([0.75, _top_lc+_lc_spany+i*(_spany+_sp_height), 0.2, _sp_height*0.95], polar=True)
            axes.append([axs, axt])
        #
        # - Plotter
        #
        # LightCurves
        lc = self.lightcurve.show(ax=axlc, 
                                  zprop=dict(ls="-", color="0.6",lw=0.5))
        redshift = self.salt2param["redshift"]
        # - No Spectra
        if not self.has_spectra():
            axs.text(0.5,0.5, f"no Spectra for {self.name}", 
                         transform=axs.transAxes,
                         va="center", ha="center")
            axs.set_yticks([]) ;axs.set_xticks([])
            clearwhich = ["left","right","top"] # "bottom"
            [axs.spines[which].set_visible(False) for which in clearwhich]
        # - Multiple spectra            
        else:
            for i,spec_ in enumerate(np.atleast_1d(self.spectra)[::-1]):
                phase = spec_.get_phase( self.salt2param["t0"] )
                label=rf"{self.name} z={redshift:.3f} | $\Delta$t: {phase:+.1f}"
                sp = spec_.show_snidresult(axes=axes[i], nbest=nbest,
                                          label=label, color_data=COLORS[i],
                                          spiderkwargs=spiderkwargs)
                # - ObsLine
                axlc.axvline(spec_.get_obsdate().datetime, 
                             ls="--", color=COLORS[i])
                if i>0:
                    axes[i][0].set_xlabel("")
                else:
                    axes[i][0].set_xlabel(axes[i][0].get_xlabel(), fontsize="small")

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
    
    def has_spectra(self):
        """ """
        return self.spectra is not None
    
    def has_multiple_spectra(self):
        """ """
        return self.nspectra>1
    
    @property
    def nspectra(self):
        """ """
        if not self.has_spectra():
            return 0
        return len(np.atleast_1d(self.spectra))
    
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
    
