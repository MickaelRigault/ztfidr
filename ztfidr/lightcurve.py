import os
import pandas
import numpy as np
import warnings
from . import io


ZTFCOLOR = { # ZTF
        "p48r":dict(marker="o",ms=7,  mfc="C3"),
        "p48g":dict(marker="o",ms=7,  mfc="C2"),
        "p48i":dict(marker="o",ms=7, mfc="C1")
}

BAD_ZTFCOLOR = { # ZTF
        "p48r":dict(marker="o",ms=6,  mfc="None", mec="C3"),
        "p48g":dict(marker="o",ms=6,  mfc="None", mec="C2"),
        "p48i":dict(marker="o",ms=6,  mfc="None", mec="C1")
}

# ================== #
#                    #
#   LIGHTCURVES      #
#                    #
# ================== #
    
class LightCurve( object ):
    """ """
    def __init__(self, data, meta=None, salt2param=None, use_dask=False):
        """ """
        self.set_data(data)
        self.set_meta(meta)
        self.set_salt2param(salt2param)
        self._use_dask = use_dask
        
    @classmethod
    def from_filename(cls, filename, use_dask=False):
        """  """
        if use_dask:
            from dask import delayed
            # This is faster than dd.read_cvs and does what we need here
            lc = delayed(pandas.read_csv)(filename,  delim_whitespace=True, comment='#')
        else:
            lc = pandas.read_csv(filename,  delim_whitespace=True, comment='#')
            
        meta = pandas.Series([os.path.basename(filename).split("_")[0]], 
                             index=["name"])
        return cls(lc, meta=meta, use_dask=use_dask)

    @classmethod
    def from_name(cls, targetname, use_dask=False, load_salt2param=True, **kwargs):
        """ """
        filename = io.get_target_lc(targetname)            
        this = cls.from_filename(filename, use_dask=use_dask, **kwargs)
        this._targetname = targetname
        if load_salt2param:
            this.load_salt2param()
            
        return this
    # ================ #
    #    Method        #
    # ================ #
    # --------- #
    #  LOADER   #
    # --------- #
    def load_salt2param(self):
        """ """
        targetname = self.targetname
        if targetname is None:
            warnings.warn("Unknown targetname (=None) ; use manually set_salt2param()")
            return None
        from .salt2 import get_target_salt2param
        salt2param = get_target_salt2param(targetname)
        self.set_salt2param(salt2param)
    # --------- #
    #  SETTER   #
    # --------- #
    def set_data(self, data):
        """ """
        self._data = data

    def set_salt2param(self, salt2param):
        """ """
        self._salt2param = salt2param
        
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
        
    def get_saltmodel(self):
        """ """
        from .salt2 import get_saltmodel
        return get_saltmodel(**self.salt2param.rename({"redshift":"z"}
                                                     )[["z","t0","x0","x1","c",
                                                        "mwebv"]].to_dict()
                            )
    def get_lcdata(self, zp=None, in_mjdrange=None, min_detection=None):
        """ """
        if zp is None:
            zp = self.data["ZP"].values
            coef = 1.
        else:
            coef = 10 ** (-(self.data["ZP"].values - zp) / 2.5)
            
        flux  = self.data["flux"] * coef
        error = self.data["flux_err"] * coef
        detection = flux/error
        
        lcdata = self.data[["mjd","mag","mag_err","filter","field_id","x_pos","y_pos", "flag","mag_lim"]].copy()
        lcdata["zp"] = zp
        lcdata["flux"] = flux
        lcdata["error"] = error
        lcdata["detection"] = detection
        lcdata["filter"] = lcdata["filter"].replace("ztfg","p48g").replace("ztfr","p48r").replace("ztfi","p48i") 

        if self.has_salt2param():
            lcdata["phase"] = lcdata["mjd"]-self.salt2param['t0']
        else:
            lcdata["phase"] = np.NaN
            
        if in_mjdrange is not None:
            lcdata = lcdata[lcdata["mjd"].between(*in_mjdrange)]

        if min_detection is not None:
            lcdata = lcdata[lcdata["detection"]>min_detection]


        return lcdata
        
    def show(self, ax=None, figsize=None, zp=None, formattime=True, zeroline=True,
                 incl_salt2=True, autoscale_salt=True, clear_yticks=True,
                 zprop={}, inmag=False, ulength=0.1, ualpha=0.1, notplt=False, **kwargs):
        """ """
        from matplotlib import dates as mdates
        from astropy.time import Time
        
        self._compute()
        # - Axes Definition
        if ax is None:
            import matplotlib.pyplot as mpl
            fig = mpl.figure(figsize=[7,4] if figsize is None else figsize)
            ax = fig.add_axes([0.1,0.15,0.8,0.75])
        else:
            fig = ax.figure
            
        # - End axes definition
        # -- 
        # - Data
        base_prop = dict(ls="None", mec="0.9", mew=0.5, ecolor="0.7", zorder=7)
        bad_prop  = dict(ls="None", mew=1, ecolor="0.7", zorder=6)        
        lineprop  = dict(color="0.7", zorder=1, lw=0.5)
        
        saltmodel = self.get_saltmodel() if incl_salt2 else None
        modeltime = self.salt2param.t0 + np.linspace(-15,50,100)
        t0 = self.salt2param.t0
        timerange = [t0-30, t0+100]
        lightcurves = self.get_lcdata(zp=zp, in_mjdrange=timerange)
        
        bands = np.unique(lightcurves["filter"])
        
        # flag goods
        flag_good = (lightcurves.flag&1==0) & (lightcurves.flag&2==0) & (lightcurves.flag&4==0) & (lightcurves.flag&8==0)

        max_saltlc = 0
        min_saltlc = 100
        # Loop over bands
        for band_ in bands:
            if band_ not in ZTFCOLOR:
                warnings.warn(f"WARNING: Unknown instrument: {band_} | magnitude not shown")
                continue

            flagband   = (lightcurves["filter"]==band_)
            
            bdata = lightcurves[flagband]
            flag_good_ = flag_good[flagband]
                
            # IN FLUX
            if not inmag:
                # - Data
                datatime = Time(bdata["mjd"], format="mjd").datetime
                y, dy = bdata["flux"], bdata["error"]
                # - Salt                
                saltdata = saltmodel.bandflux(band_, modeltime, zp=self.flux_zp, zpsys="ab") if saltmodel is not None else None
                    
            # IN MAG                                
            else:
                flag_det = (lightcurves["mag"]<99)
                # - Data                
                bdata = bdata[flag_det]
                flag_good_ = flag_good_[flag_det]
                datatime = Time(bdata["mjd"], format="mjd").datetime
                y, dy = bdata["mag"], bdata["mag_err"]
                # - Salt
                saltdata = saltmodel.bandmag(band_, "ab",modeltime) if saltmodel is not None else None
                
            # -> good
            ax.errorbar(datatime[flag_good_],
                            y[flag_good_],  yerr=dy[flag_good_], 
                            label=band_, 
                            **{**base_prop, **ZTFCOLOR[band_],**kwargs}
                            )
            # -> bad
            ax.errorbar(datatime[~flag_good_],
                            y[~flag_good_],  yerr=dy[~flag_good_], 
                            label=band_, 
                            **{**bad_prop, **BAD_ZTFCOLOR[band_],**kwargs}
                            )
            ax.plot(Time(modeltime, format="mjd").datetime,
                    saltdata,
                    color=ZTFCOLOR[band_]["mfc"], zorder=5)
            
            max_saltlc = np.max([max_saltlc, np.max(saltdata)])
            min_saltlc = np.min([min_saltlc, np.min(saltdata)])
            
        if inmag:
            ax.invert_yaxis()
            for band_ in bands:
                bdata = lightcurves[(lightcurves["filter"]==band_) & (lightcurves["mag"]>=99)]
                datatime = Time(bdata["mjd"], format="mjd").datetime
                y = bdata["mag_lim"]
                ax.errorbar(datatime, y,
                                 yerr=ulength, lolims=True, alpha=ualpha,
                                 color=ZTFCOLOR[band_]["mfc"], 
                                 ls="None",  label="_no_legend_")
                                 
                                 
                
        if formattime:
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
                ax.set_yticklabels(["" for _ in ax.get_yticklabels()])
            
        if autoscale_salt:
            ax.set_xlim(*Time(timerange,format="mjd").datetime)
            if not inmag:
                ax.set_ylim(bottom=-max_saltlc*0.25)
                ax.set_ylim(top=max_saltlc*1.25)
            else:
                print(min_saltlc)
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
    

    def has_salt2param(self):
        """" """
        return self.salt2param is not None
    
    @property
    def salt2param(self):
        """ """
        if not hasattr(self,"_salt2param"):
            return None
        return self._salt2param


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
