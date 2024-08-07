import os
import pandas
import numpy as np
import warnings
from . import io


__all__ =["get_target_lightcurve"]



ZTFCOLOR = { # ZTF
        "ztfr":dict(marker="o",ms=7,  mfc="C3"),
        "ztfg":dict(marker="o",ms=7,  mfc="C2"),
        "ztfi":dict(marker="o",ms=7, mfc="C1")
}

BAD_ZTFCOLOR = { # ZTF
        "ztfr":dict(marker="o",ms=6,  mfc="None", mec="C3"),
        "ztfg":dict(marker="o",ms=6,  mfc="None", mec="C2"),
        "ztfi":dict(marker="o",ms=6,  mfc="None", mec="C1")
}


def get_target_lcdata(name, phase_range=None, mjd_range=None, saltparam=None, **kwargs):
    """ load the lightcurve object for the given target. """
    lc = LightCurve.from_name(name, saltparam=saltparam, **kwargs)
    data = lc.get_lcdata(**kwargs)[["mjd", "flux", "error", "phase", "filter", "flag", "field_id",  "rcid"]].copy() # "x_pos", "y_pos",
    # phase means restframe, so old phase renamed phase_obsframe
    data["phase_obsframe"] = data["phase"].copy()
    data["phase"] = data["phase_obsframe"]/(1+saltparam["redshift"])
    
    if phase_range is not None:
        data = data[data["phase_restframe"].between(*phase_range)]

    if mjd_range is not None:
        data = data[data["mjd"].between(*mjd_range)]

    return data


def get_target_lcresiduals(name, phase_range=None, mjd_range=None, saltparam=None,
                               which_model=None, **kwargs):
    """ shortcut to get the target lc residuals """
    data = LightCurve.from_name(name, saltparam=saltparam).get_model_residual(which=which_model, **kwargs)
    if phase_range is not None:
        data = data[data["phase"].between(*phase_range)]

    if mjd_range is not None:
        data = data[data["mjd"].between(*mjd_range)]

    return data

def get_saltmodel(which="salt2.4", **params):
    """ """
    import sncosmo    
    if which is None or which in ["salt2.4"]:
        which = "salt2 v=2.4"
    
    # parsing model version
    source_name, *version = which.split("v=")
    source_name = source_name.strip()
    version = None if len(version)==0 else version[0].strip()
    source = sncosmo.get_source(source_name, version=version, copy=True)
    
    dust  = sncosmo.CCM89Dust()
    model = sncosmo.Model(source, effects=[dust],
                              effect_names=['mw'],
                              effect_frames=['obs'])
    model.set(**params)
    return model

# ================== #
#                    #
#   LIGHTCURVES      #
#                    #
# ================== #
    
class LightCurve( object ):

    def __init__(self, data, meta=None, saltparam=None, saltmodel=None, use_dask=False):
        """ likely, this is not how you should load the data. 
        See from_name() or from_filename() class methods.
        """
        self.set_data(data)
        self.set_meta(meta)
        self.set_saltparam(saltparam)
        self._use_dask = use_dask
        self._saltmodel = saltmodel
        
    @classmethod
    def from_filename(cls, filename, saltparam=None, use_dask=False, saltmodel=None, **kwargs):
        """  load a Lightcurve object from a given file. """
        if use_dask:
            from dask import delayed
            # This is faster than dd.read_cvs and does what we need here
            lc = delayed(pandas.read_csv)(filename,  delim_whitespace=True, comment='#')
        else:
            lc = pandas.read_csv(filename,  delim_whitespace=True, comment='#')
            
        meta = pandas.Series([os.path.basename(filename).split("_")[0]], 
                             index=["name"])
        return cls(lc, meta=meta, use_dask=use_dask,
                       saltparam=saltparam, saltmodel=saltmodel,
                       **kwargs)

    @classmethod
    def from_name(cls, targetname, use_dask=False, saltparam=None, saltmodel=None, **kwargs):
        """ """
        filename = io.get_target_lightcurve(targetname, load=False)
        this = cls.from_filename(filename, use_dask=use_dask, saltparam=saltparam, saltmodel=saltmodel, **kwargs)
        this._targetname = targetname            
        return this
    
    # ================ #
    #    Method        #
    # ================ #
    # --------- #
    #  LOADER   #
    # --------- #

    # --------- #
    #  SETTER   #
    # --------- #
    def set_data(self, data):
        """ """
        self._data = data

    def set_saltparam(self, saltparam):
        """ """
        self._saltparam = saltparam
        
    def set_meta(self, meta):
        """ """
        self._meta = meta  

    # --------- #
    #  GETTER   #
    # --------- #
    def get_obsphase(self, min_detection=5, groupby=None, **kwargs):
        """ 
        Returns
        -------
        pandas.Series 
        """
        lcdata = self.get_lcdata(min_detection=min_detection, **kwargs)
        if groupby is None:
            return lcdata["phase"]
        
        return lcdata.groupby(groupby)["phase"].apply( list )
        
    def get_saltmodel(self, which=None):
        """ """
        if which is None:
            which = self._saltmodel
        propmodel = self.saltparam.rename({"redshift":"z"})[["z","t0","x0","x1","c","mwebv"]].to_dict()
        return get_saltmodel(which=which, **propmodel)
    
    def get_lcdata(self, zp=None, in_mjdrange=None,
                       min_detection=None,
                       filters=None,
                       flagout=[1,2,4,8,16]):
        """ 
        filters: [string, None or list]
            list of filters 
            - None/'*' or 'all': no filter selection/
            - string: just this filter (e.g. 'ztfg')
            - list of string: just these filters (e.g. ['ztfg','ztfr'])

        flagout: [list of int or string]
            flag == 0 means all good, but in details:
            
            0: no warning 
            1: flux_err==0 Remove unphysical errors 
            2: chi2dof>3: Remove extreme outliers 
            4: cloudy>1: BTS cut 
            8: infobits>0: BTS cut 16: mag_lim<19.3: Cut applied in Dhawan 2021 
            32: seeing>3: Cut applied in Dhawan 2021 
            64: fieldid>879: Recommended IPAC cut 
            128: moonilf>0.5: Recommended IPAC cut 
            256: has_baseline>1: Has a valid baseline correction 
            512: airmass>2: Recommended IPAC cut 
            1024: flux/flux_err>=5: Nominal detection

        """
        from .utils import flux_to_mag

        if flagout in ["all","any","*"]:
            data = self.data[self.data["flag"]==0]
            
        elif flagout is None:
            data = self.data.copy()
        else:
            flag_ = np.all([(self.data.flag&i_==0) for i_ in np.atleast_1d(flagout)], axis=0)
            data = self.data[flag_]
            

        if zp is None:
            zp = data["ZP"].values
            coef = 1. 
        else:
            coef = 10 ** (-(data["ZP"].values - zp) / 2.5)

        flux  = data["flux"] * coef
        error = data["flux_err"] * coef
        detection = flux/error
            
        
        lcdata = data[["mjd","mag","mag_err","filter","field_id", "flag", "rcid"]] # "x_pos","y_pos"
        additional = pandas.DataFrame(np.asarray([zp, flux, error, detection]).T,
                                         columns=["zp", "flux", "error", "detection"],
                                         index=lcdata.index)
        
        additional["mag_lim"], _ = flux_to_mag(error*5, None, zp=zp)
        
        lcdata = pandas.merge(lcdata, additional, left_index=True, right_index=True)
#        lcdata.loc["zp",:] = zp
#        lcdata["flux"] = flux
#        lcdata["error"] = error
#        lcdata["detection"] = detection
        lcdata["filter"] = lcdata["filter"].replace("ztf","ztf")
        
        
        if self.has_saltparam():
            lcdata["phase"] = lcdata["mjd"]-self.saltparam['t0']
        else:
            lcdata["phase"] = np.NaN
            
        if in_mjdrange is not None:
            lcdata = lcdata[lcdata["mjd"].between(*in_mjdrange)]

        if min_detection is not None:
            lcdata = lcdata[lcdata["detection"]>min_detection]

        if filters is not None and filters not in ["*","all"]:
            lcdata = lcdata[lcdata["filter"].isin(np.atleast_1d(filters))]
            
        return lcdata

    def get_model_residual(self, model=None, which=None,
                           intrinsic_error=None, 
                           **kwargs):
        """ get a dataframe with lightcurve data, model and residuals information.

        Parameters
        ----------
        model: [sncosmo.Model or None] -optional-
            provide the sncosmo model from which the model flux can be obtained.
            If None given [default] this method will call self.get_saltmodel().

        modelprop: [dict] -optional-
            kwarg information passed to model.set() to change the default model parameters.
            = This is only used in model=None = 
            It aims at updating model given by self.get_saltmodel()
            
        intrinsic_error: [float] -optional-
            provide an intrinsic error for the lightcurve. This will be stored as 'error_int'
            in the returned DataFrame. 
            The 'pull' will use the quadratic sum of error and error_int to be calculated.
            if None given [default] this will assume 0.

        
        **kwargs goes to get_lcdata()

        Returns
        -------
        DataFrame
        """
        basedata = self.get_lcdata(**kwargs)[["mjd","flux", "error","phase", "filter","flag", "field_id", "rcid"]].copy() # "x_pos", "y_pos",
        if model is None:
            model = self.get_saltmodel(which)

        # Model
        basedata["model"] = model.bandflux(basedata["filter"], basedata["mjd"], 
                                            zp=self.flux_zp, zpsys="ab")
        # Residual    
        basedata["residual"] = basedata["flux"] - basedata["model"]

        # Error    
        basedata["error_int"] = intrinsic_error if intrinsic_error is not None else 0
        total_error = np.sqrt(basedata["error"]**2 + basedata["error_int"]**2)
        # Pull    
        basedata["pull"] = basedata["residual"]/total_error
        return basedata
    
    def get_sncosmotable(self, min_detection=5,  phase_range=[-10,30], filters=["ztfr","ztfg"], **kwargs):
        """ """
        from .utils import mag_to_flux
        
        t0 = self.saltparam["t0"]
        to_fit = self.get_lcdata(min_detection=min_detection, in_mjdrange= t0 + phase_range,
                                 filters=filters, **kwargs)
        sncosmo_lc = to_fit.rename({"mjd":"time", "filter":"band"}, axis=1)[["time","band","zp","mag","mag_err"]]
        sncosmo_lc["zpsys"] = "ab"
        sncosmo_lc["flux"], sncosmo_lc["flux_err"] = mag_to_flux(sncosmo_lc["mag"],
                                                                 sncosmo_lc["mag_err"],
                                                                 zp=sncosmo_lc["zp"])
        return sncosmo_lc

    def fit_salt(self, free_parameters=['t0', 'x0', 'x1', 'c'],
                       min_detection=5, phase_range=[-10,30], filters=["ztfr","ztfg"],
                       as_dataframe=False, which=None, modelprop={},
                       **kwargs):
        """ """
        import sncosmo
        from astropy import table
        from .salt2 import salt2result_to_dataframe

        model = self.get_saltmodel(which=which)
        
        if modelprop:
            model.set( **modelprop )
            print(dict(zip(model.param_names, model.parameters)))
            
        sncosmo_df = self.get_sncosmotable(min_detection=min_detection, 
                                           phase_range=phase_range, 
                                           filters=filters)
        
        fitted_data = table.Table.from_pandas(sncosmo_df)
        (result, fitted_model) = sncosmo.fit_lc(fitted_data, model,
                                                vparam_names=free_parameters,
                                                **kwargs)
         
        if as_dataframe:
            return salt2result_to_dataframe(result)
        
        return (fitted_data,model), (result, fitted_model)

    def fit_salt_perfilter(lc, filters=["ztfg",'ztfr','ztfi'],
                           min_detection=5, free_parameters=['t0','x0', 'x1'],
                           phase_range=[-10, 30], t0_range=[-2,+2]):
        """ """
        results = []
        for filter_ in filters:
            try:
                result  = lc.fit_salt(min_detection=min_detection,
                                      filters=filter_,
                                      free_parameters=free_parameters,
                                      phase_range=phase_range,
                                      bounds={"t0":lc.saltparam['t0']+t0_range},
                                      as_dataframe=True
                                      )
            except:
                warnings.warn(f"failed for filter {filter_}")
                result = pandas.DataFrame()

            results.append(result)
            
        return pandas.concat(results, keys=filters)
        
    # --------- #
    #  PLOTTER  #
    # --------- #
    def show(self, ax=None, figsize=None, zp=None, bands="*",
                 formattime=True, zeroline=True,
                 incl_salt=True, which_model=None, autoscale_salt=True, clear_yticks=True,
                 phase_range=[-30,100], as_phase=False, t0=None, 
                 zprop={}, inmag=False, ulength=0.1, ualpha=0.1, notplt=False,
                 rm_flags=True, **kwargs):
        """ """
        from matplotlib import dates as mdates
        from astropy.time import Time
        
        self._compute()
        # - Axes Definition
        if ax is None:
            import matplotlib.pyplot as mpl
            fig = mpl.figure(figsize=[7,4])# if figsize is None else figsize)
            ax = fig.add_axes([0.1,0.15,0.8,0.75])
        else:
            fig = ax.figure
            
        # - End axes definition
        # -- 
        # - Data
        base_prop = dict(ls="None", mec="0.9", mew=0.5, ecolor="0.7", zorder=7)
        bad_prop  = dict(ls="None", mew=1, ecolor="0.7", zorder=6)        
        lineprop  = dict(color="0.7", zorder=1, lw=0.5)

        if incl_salt:
            saltmodel = self.get_saltmodel(which=which_model)
        else:
             saltmodel = None
             autoscale_salt = False

             
        t0 = self.saltparam.t0
        if not np.isnan(t0):
            if phase_range is not None: # removes NaN
                timerange = [t0+phase_range[0], t0+phase_range[1]]
            else:
                timerange = None
                
            modeltime = t0 + np.linspace(-15,50,100)
        else:
            timerange = None
            if incl_salt:
                warnings.warn("t0 in saltparam is NaN, cannot show the model")
            if as_phase:
                warnings.warn("t0 in saltparam is NaN, as_phase not available")
                as_phase = False
                
            incl_salt = False
            saltmodel = None
            autoscale_salt = False
            
        if not rm_flags:
            prop = {"flagout":None}
        else:
            prop = {}
        lightcurves = self.get_lcdata(zp=zp, in_mjdrange=timerange, **prop)
        if bands is None or bands in ["*", "all"]:
            bands = np.unique(lightcurves["filter"])
        else:
            bands = np.atleast_1d(bands)
        

        max_saltlc = 0
        min_saltlc = 100
        # Loop over bands
        for band_ in bands:
            if band_ not in ZTFCOLOR:
                warnings.warn(f"WARNING: Unknown instrument: {band_} | magnitude not shown")
                continue

            flagband   = (lightcurves["filter"]==band_)
            
            bdata = lightcurves[flagband]
#            flag_good_ = flag_good[flagband]
            
            # IN FLUX
            if not inmag:
                # - Data
                if as_phase:
                    datatime = bdata["mjd"].astype("float") - t0
                else:
                    datatime = Time(bdata["mjd"].astype("float"), format="mjd").datetime
                    
                y, dy = bdata["flux"], bdata["error"]
                # - Salt
                if saltmodel is not None:
                    saltdata = saltmodel.bandflux(band_, modeltime, zp=self.flux_zp, zpsys="ab") \
                      if saltmodel is not None else None
                else:
                    saltdata = None
                    
            # IN MAG                                
            else:
                flag_det = (lightcurves["mag"]<99)
                # - Data                
                bdata = bdata[flag_det]
                #flag_good_ = flag_good_[flag_det]
                if as_phase:
                    datatime = bdata["mjd"].astype("float") - t0
                else:
                    datatime = Time(bdata["mjd"], format="mjd").datetime
                    
                y, dy = bdata["mag"], bdata["mag_err"]
                # - Salt
                if saltmodel is not None:
                    saltdata = saltmodel.bandmag(band_, "ab",modeltime) if saltmodel is not None else None
                else:
                    saltdata = None
            
            # -> good
            ax.errorbar(datatime,#[flag_good_],
                            y,#[flag_good_],
                            yerr=dy,#[flag_good_], 
                            label=band_, 
                            **{**base_prop, **ZTFCOLOR[band_],**kwargs}
                            )
            # -> bad
            ax.errorbar(datatime,#[~flag_good_],
                            y,#[~flag_good_],
                            yerr=dy,#[~flag_good_], 
                            label=band_, 
                            **{**bad_prop, **BAD_ZTFCOLOR[band_],**kwargs}
                            )
        
            if saltdata is not None:
                if as_phase:
                    modeltime_ = modeltime - t0
                else:
                    modeltime_ = Time(modeltime, format="mjd").datetime
                    
                ax.plot(modeltime_,
                        saltdata,
                        color=ZTFCOLOR[band_]["mfc"], zorder=5)

                max_saltlc = np.max([max_saltlc, np.max(saltdata)])
                min_saltlc = np.min([min_saltlc, np.min(saltdata)])
            
        if inmag:
            ax.invert_yaxis()
            for band_ in bands:
                bdata = lightcurves[(lightcurves["filter"]==band_) & (lightcurves["mag"]>=99)]
                if as_phase:
                    datatime = Time(bdata["mjd"], format="mjd").datetime
                else:
                    datatime = bdata["mjd"].astype("float") - t0
                    
                y = bdata["mag_lim"]
                ax.errorbar(datatime, y,
                                 yerr=ulength, lolims=True, alpha=ualpha,
                                 color=ZTFCOLOR[band_]["mfc"], 
                                 ls="None",  label="_no_legend_")
                                 
        if formattime and not as_phase:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        lunit = "flux" if not inmag else "mag"
        ax.set_ylabel(f"{lunit} [zp={zp}]" if zp is not None else f"{lunit} []")
        if zeroline:
            ax.axhline(0 if not inmag else 22, **{**dict(color="0.7",ls="--",lw=1, zorder=1),**zprop} )

        if not inmag:
            max_data = np.percentile(lightcurves["flux"], 99.)
            mean_error = np.nanmean(lightcurves["error"])
            ax.set_ylim(-2*mean_error, max_data*1.15)
            if clear_yticks:
                ax.axes.yaxis.set_ticklabels([])
            
        if autoscale_salt:
            if timerange is not None:
                if as_phase:
                    ax.set_xlim(*(np.asarray(timerange)-t0))
                else:
                    ax.set_xlim(*Time(timerange,format="mjd").datetime)
                    
            if not inmag:
                ax.set_ylim(bottom=-max_saltlc*0.25)
                ax.set_ylim(top=max_saltlc*1.25)
            else:
                if np.isinf(min_saltlc) or np.isnan(min_saltlc):
                    ax.set_ylim(23, 14)
                else:
                    ax.set_ylim(top=min_saltlc*0.95)
                    
        return fig
    
    # ----------- #
    #  Internal   #
    # ----------- #
    def _compute(self, persist=False):
        """ """
        if not self._use_dask:
            return 
        
        from dask.dataframe.core import DataFrame as daskDataFrame
        from dask.delayed import Delayed
        if not type(self._data) in [daskDataFrame, Delayed]:
            return
        if persist:
            self._data = self._data.persist()
        else:
            self._data = self._data.compute()
    
    # ================ #
    #   Properties     #
    # ================ #        
    # Baseline    
    @property
    def data(self):
        """ """
        return self._data
    
    @property
    def meta(self):
        """ """
        return self._meta

    @property
    def targetname(self):
        """ """
        if not hasattr(self, "_targetname"):
            if self.meta is not None:
                return self.meta.get("name",None)
            
            return None
        
        return self._targetname
    
    def has_saltparam(self):
        """" """
        return self.saltparam is not None
    
    @property
    def saltparam(self):
        """ """
        if not hasattr(self,"_saltparam"):
            return None
        return self._saltparam

    @property
    def flux_zp(self):
        """ """
        zp = self.data["ZP"]
        if len(np.unique(zp)) == 1:
            return float(zp[0])
        return zp
    
class LightCurveCollection( object ):
    
    def __init__(self, lightcurves, use_dask=False):
        """ """
        self.set_lightcurves(lightcurves)
        self._use_dask = use_dask
            
    @classmethod
    def from_directory(cls, directory, contains=None, startswith=None, 
                       extension=".csv", use_dask=False):
        """ """
        from glob import glob
        glob_format = "*" if not startswith else f"{startswith}*"
        if contains is not None:
            glob_format += f"{contains}*"
        if extension is not None:
            glob_format +=f"{extension}"
            
        files = glob(os.path.join(directory, glob_format))
        
        return cls.from_filenames(files, use_dask=use_dask)
    
    @classmethod
    def from_filenames(cls, filenames, use_dask=False, **kwargs):
        """ """
        return cls([LightCurve.from_filename(file_, use_dask=use_dask, **kwargs) 
                    for file_ in filenames], use_dask=use_dask)
        
    # ---------------- #
    #   Internal       #
    # ---------------- #
    def _persist(self):
        """ """
        if not self._use_dask:
            return 
        
        _ = [lc._compute(persist=True) for lc in self.lightcurves]
        
    # ================ #
    #    Method        #
    # ================ #
    def set_lightcurves(self, lightcurves):
        """ """
        self._lightcurves = lightcurves
    
    def get_meta(self, rebuild=False):
        """ """
        if rebuild:
            metas = self.call_down("meta", False)
            return pandas.DataFrame(metas)

        return self.meta

    def get_target_lightcurve(self, name):
        """ """
        lc_ = np.asarray(self.lightcurves)[self.meta.query(f"name == '{name}'").index]
        if len(lc_)==1:
            return lc_[0]

        return lc_

    def show_target(self, name, ax=None, **kwargs):
        """ """
        lctarget = self.get_target_lightcurve(name)
        return lctarget.show(ax=ax, **kwargs)
    
    # -------- #
    # INTERNAL #
    # -------- #
    def map_down(self, what, margs, *args, **kwargs):
        """ """
        return [getattr(spec, what)(marg, *args, **kwargs)
                for lc, marg in zip(self.lightcurves, margs)]
    
    def call_down(self, what, isfunc, *args, **kwargs):
        """ """
        if isfunc:
            return [getattr(lc,what)(*args, **kwargs) 
                    for lc in self.lightcurves]
        return [getattr(lc,what) for lc in self.lightcurves]    
    
    
    @property
    def meta(self):
        """ """
        if not hasattr(self,"_meta"):
            self._meta = self.get_meta(rebuild=True)
        return self._meta
    
    # Baseline    
    @property
    def lightcurves(self):
        """ """    
        return self._lightcurves
