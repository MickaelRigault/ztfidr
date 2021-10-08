import os
import pandas
import numpy as np

ZTFCOLOR = { # ZTF
        "p48r":dict(marker="o",ms=7,  mfc="C3"),
        "p48g":dict(marker="o",ms=7,  mfc="C2"),
        "p48i":dict(marker="o",ms=7, mfc="C1")
}

class LightCurve( object ):
    """ """
    def __init__(self, data, meta=None, use_dask=False):
        """ """
        self.set_data(data)
        self.set_meta(meta)
        self._use_dask = use_dask
        
    @classmethod
    def from_filename(cls, filename, use_dask=False):
        """  """
        if use_dask:
            from dask import delayed
            # This is faster than dd.read_cvs and does what we need here
            lc = delayed(pandas.read_csv)(filename, index_col=0)
        else:
            lc = pandas.read_csv(filename, index_col=0)
        meta = pandas.Series([os.path.basename(filename).split("_")[0]], 
                             index=["name"])
        return cls(lc, meta=meta, use_dask=use_dask)
    
    # ================ #
    #    Method        #
    # ================ #
    def set_data(self, data):
        """ """
        self._data = data
        
    def set_meta(self, meta):
        """ """
        self._meta = meta  
        
    def get_lcdata(self, zp=25):
        """ """
        if zp is None:
            zp = self.data["magzp"].values
            coef = 1.
        else:
            coef = 10 ** (-(self.data["magzp"].values - zp) / 2.5)
            
        flux  = self.data["ampl"] * coef
        error = self.data["ampl.err"] * coef
        detection = flux/error
        
        lcdata = self.data[["obsmjd", "filterid", "fieldid","target_x","target_y"]].copy()
        lcdata["zp"] = zp
        lcdata["flux"] = flux
        lcdata["error"] = error
        lcdata["detection"] = detection
        lcdata["filter"] = lcdata["filterid"].replace(1,"p48g").replace(2,"p48r").replace(3,"p48i")
        return lcdata
        
    def show(self, ax=None, zp=25, formattime=True, zeroline=True, zprop={}, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        from matplotlib import dates as mdates
        from astropy.time import Time
        
        self._compute()
        # - Axes Definition
        if ax is None:
            fig = mpl.figure(figsize=[7,4])
            ax = fig.add_axes([0.1,0.15,0.8,0.75])
        else:
            fig = ax.figure
            
        # - End axes definition
        # -- 
        # - Data
        base_prop = dict(ls="None", mec="0.9", mew=0.5, ecolor="0.7")
        lineprop = dict(color="0.7", zorder=1, lw=0.5)
        
        lightcurves = self.get_lcdata(zp=zp)
        bands = np.unique(lightcurves["filter"])
        
        # Loop over bands
        for band_ in bands:
            if band_ not in ZTFCOLOR:
                warnings.warn(f"WARNING: Unknown instrument: {band_} | magnitude not shown")
                continue
            
            bdata = lightcurves[lightcurves["filter"]==band_]
            datatime = Time(bdata["obsmjd"], format="mjd").datetime
            y, dy = bdata["flux"], bdata["error"]
            ax.errorbar(datatime,
                         y,  yerr= dy, 
                         label=band_, 
                         **{**base_prop, **ZTFCOLOR[band_],**kwargs}
                       )
        if formattime:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        ax.set_ylabel(f"Flux [zp={zp}]" if zp is not None else "Flux []")
        if zeroline:
            ax.axhline(0, **{**dict(color="0.7",ls="--",lw=1, zorder=1),**zprop} )
            
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
