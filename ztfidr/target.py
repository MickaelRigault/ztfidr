import pandas
import numpy as np
import warnings


from . import io

TARGET_DATA = io.get_targets_data()

COLORS = ["0.6"]*10 # , "tan", "lightsteelblue", "thistle", "darkseagreen"]

class Target():
    """ """
    def __init__(self, lightcurve, spectra, meta=None):
        """ """
        self.set_lightcurve(lightcurve)
        self.set_spectra(spectra)
        self.set_meta(meta)
        
    @classmethod
    def from_name(cls, targetname, load_snidres=True):
        """ """
        from . import lightcurve, spectroscopy
        lc = lightcurve.LightCurve.from_name(targetname)
        spec = spectroscopy.Spectrum.from_name(targetname, as_spectra=False, load_snidres=load_snidres)
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
        
    def set_spectra(self, spectra):
        """ """
        self._spectra = spectra   
    
    def set_meta(self, meta):
        """ """
        self._meta = meta
        
    # -------- #
    #  GETTER  #
    # -------- #
    def get_autotyping(self, full_output=True, remove_fullna=True, allow_snidrun=True):
        """ """
        notyping = pandas.Series([np.NaN]*4, index=['type', 'subtype', 'p(type)', 'p(subtype|type)'])
        if self.nspectra == 0:
            return notyping
                
        types = np.asarray([sn_.get_type() if sn_ is not None else ((np.NaN,np.NaN),(np.NaN, np.NaN))
                            for sn_ in np.atleast_1d(self.get_snidresult(allow_run=allow_snidrun))
                            ]).T
        
        phases = [np.round(spec_.get_phase(self.salt2param['t0']), 1) for spec_ in np.atleast_1d(self.spectra)]
        df_types = pandas.DataFrame(np.concatenate(types,axis=0).T,
                    columns=["type","subtype","p(type)","p(subtype|type)"], 
                    index=pandas.Series(phases, name="phase")).replace({"unclear":np.NaN,"nan":np.NaN})

        
        if remove_fullna:
            df_types = df_types[~df_types.isna().all(axis=1)]

        if full_output:
            return df_types

        df_types = df_types.drop_duplicates()


        if len(df_types)==0:
            return notyping

        if len(df_types)==1:
            return df_types.iloc[0]

        types = df_types["type"].unique()
        if len(types)==1:
            typing = {"type":types[0], "p(type)":df_types["p(type)"].astype(float).max()}
            subtypes = df_types["subtype"].unique()
            if len(subtypes) == 1:
                typing["subtype"] = subtypes[0]
                typing["p(subtype|type)"] = df_types["p(subtype|type)"].astype(float).max()
            else:
                typing["subtype"] = "unclear"
                typing["p(subtype|type)"] = np.NaN

        else:
            typing = {"type":"unclear",
                      "p(type)":np.NaN,
                      "subtype": "unclear",
                      "p(subtype|type)":np.NaN
                     }
        return pandas.Series(typing)[['type', 'subtype', 'p(type)', 'p(subtype|type)']]
        return df_types

    def fit_snid(self, delta_phase=5, lbda_range=[4000, 8000],
                     redshift=None, set_it=True, use_phase=True,
                     **kwargs):
        """ fit SNID constraining redshift and phase """

        # Redshift input
        # Phase
        snidres = []
        for spec_ in np.atleast_1d(self.spectra):
            if use_phase:
                phase = spec_.get_phase( self.meta["t0"] )
            else:
                phase = None
                
            snidres_ = spec_.fit_snid(phase=phase, redshift=redshift,
                                      delta_phase=delta_phase, lbda_range=lbda_range)
            if set_it:
                spec_.set_snidresult(snidres_)
            
            snidres.append(snidres_)
            
        return snidres[0] if not self.has_multiple_spectra() else snidres
        
    
    def get_snidresult(self, redshift=None, zquality=2, set_it=True, allow_run=True, **kwargs):
        """ """
        if not self.has_spectra():
            warnings.warn("No spectra loaded.")
            return None
        
        if not allow_run:
            snidres = [spec_.snidresult for spec_ in np.atleast_1d(self.spectra)]
            return snidres[0] if not self.has_multiple_spectra() else snidres
        
        if redshift is None:
            z_, z_quality_ = self.get_redshift()
            if zquality is not None and \
               zquality not in ["*","all"] and \
               z_quality_ in np.atleast_1d(zquality):
                redshift = z_
            
        snidres = []
        for spec_ in np.atleast_1d(self.spectra):
            phase = spec_.get_phase( self.salt2param["t0"] )
            if spec_.snidresult is None:
                snidres_ = spec_.fit_snid(phase=phase, redshift=redshift, **kwargs)
                if set_it:
                    spec_.set_snidresult(snidres_)
            else:
                snidres_ = spec_.snidresult
                
            snidres.append(snidres_)
                
        return snidres[0] if not self.has_multiple_spectra() else snidres

    def get_redshift(self ):
        """ """
        return self.meta["redshift"], self.meta["source"]

    def get_obsphase(self, **kwargs):
        """ """
        phase_ = self.get_lc_obsphase(**kwargs)
        phase_["spectra"] = self.get_spec_obsphase()
        return phase_
    
    def get_lc_obsphase(self, groupby="filter", min_detection=5, **kwargs):
        """ """
        return self.lightcurve.get_obsphase(groupby=groupby, min_detection=min_detection, **kwargs)

    def get_spec_obsphase(self):
        """ """
        return [spec_.get_phase(self.salt2param["t0"], self.get_redshift()[0])
                                    for spec_ in np.atleast_1d(self.spectra)]
    
    # -------- #
    #  LOADER  #
    # -------- #
    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, nbest=3, lines={"Ha":6563},
                 line_color="0.6", allow_snidrun=False):
        """ """
        import matplotlib.pyplot as mpl
        n_speclines = np.max([1,self.nspectra])
        fig = mpl.figure(figsize=[7.2,3+2.5*n_speclines])
        
        _ = self.get_snidresult(allow_run=allow_snidrun)
        # - Axes
        _lc_height = 0.25/np.sqrt(n_speclines)
        _sp_height = 0.01+0.40/n_speclines
        _spany = 0.04+0.08/np.sqrt(n_speclines)
        _lc_spany = 0.03+0.11/np.sqrt(n_speclines)
        _bottom_lc = 0.04+0.05/np.sqrt(n_speclines)
        _top_lc = _bottom_lc+_lc_height

        axlc = fig.add_axes([0.15, _bottom_lc, 0.78, _lc_height])
        axes = []
        for i in range(n_speclines):
            axs = fig.add_axes([0.15, _top_lc+_lc_spany+i*(_spany+_sp_height), 0.55, _sp_height])
            
            axt = fig.add_axes([0.75, _top_lc+_lc_spany+i*(_spany+_sp_height), 0.2, _sp_height*0.8])
            axtt = fig.add_axes([0.75, _top_lc+_lc_spany+i*(_spany+_sp_height) + _sp_height*0.9, 0.2, _sp_height*0.1])
            axes.append([axs, [axtt,axt]])

        #
        # - Plotter
        #
        # LightCurves
        lc = self.lightcurve.show(ax=axlc, 
                                  zprop=dict(ls="-", color="0.6",lw=0.5))
        redshift, redshift_source = self.get_redshift()
        redshift_label = redshift_source
        
        # - No Spectra
        if not self.has_spectra():
            axs.text(0.5,0.5, f"no Spectra for {self.name}", 
                         transform=axs.transAxes,
                         va="center", ha="center")
            axs.set_yticks([]) ;axs.set_xticks([])
            clearwhich = ["left","right","top"] # "bottom"
            [axs.spines[which].set_visible(False) for which in clearwhich]
            
        # - spectra            
        else:
            for i, spec_ in enumerate(np.atleast_1d(self.spectra)[::-1]):
                phase = spec_.get_phase( self.salt2param["t0"], z=redshift)

                label=rf"{self.name} z={redshift:.3f} ({redshift_label}) | $\Delta$t: {phase:+.1f}"
                sp = spec_.show_snidresult(axes=axes[i], nbest=nbest,
                                          label=label, color_data=COLORS[i])
                # - ObsLine
                axlc.axvline(spec_.get_obsdate().datetime, 
                             ls="--", color=COLORS[i])
                if i>0:
                    axes[i][0].set_xlabel("")
                else:
                    axes[i][0].set_xlabel(axes[i][0].get_xlabel(), fontsize="medium")

                if lines is not None:
                    for line, lbda in lines.items():
                        for lbda_ in np.atleast_1d(lbda):
                            axes[i][0].axvline(lbda_*(1+redshift), ls="--",
                                           color=line_color, zorder=1, lw=0.5, alpha=0.8)

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
    
