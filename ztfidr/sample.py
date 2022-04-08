
import warnings
import numpy as np
import pandas
from . import io

class Sample():
    
    def __init__(self, salt2param=None, targetsdata=None):
        """ """
        self.set_salt2param(salt2param)
        self.set_targetsdata(targetsdata)
    
    @classmethod
    def load(cls, default_salt2=True):
        """ """
        salt2param = io.get_salt2params(default=default_salt2)
        targetsdata = io.get_targets_data()
        return cls(salt2param=salt2param, targetsdata=targetsdata)
    
    def load_fiedid(self):
        """ """
        from ztfquery import fields
        fieldid = [fields.get_fields_containing_target(s_["ra"],s_["dec"])
                       for i_,s_ in self.data.iterrows()]
        self.data["fieldid"] = fieldid


    def load_phasedf(self, min_detection=5, groupby='filter', client=None, rebuild=False, **kwargs):
        """ """
        if not rebuild:
            phasedf = io.get_phase_coverage(load=True, warn=False)
            if phasedf is None:
                rebuild = True
                
        if rebuild:
            phasedf = self.build_phase_coverage(min_detection=min_detection,
                                                    groupby=groupby,
                                                    client=client, store=True, **kwargs)
        self._phasedf = phasedf
        
    # ------- #
    # SETTER  #
    # ------- #
    def set_salt2param(self, salt2param):
        """ """
        salt2param["t0day"] = np.asarray(salt2param["t0"].astype("float"), dtype="int")
        self._salt2param = salt2param
        
    def set_targetsdata(self, targetsdata):
        """ """
        self._targetsdata = targetsdata

    def merge_to_data(self, dataframe, how="outer", **kwargs):
        """ """
        self._data = pandas.merge(self.data, dataframe, how=how, 
                                 **{**dict(left_index=True, right_index=True),
                                 **kwargs})
        
    # ------- #
    # GETTER  #
    # ------- #
    def get_data(self, clean_t0nan=True, t0range=None,
                     z_quality=None, query=None, in_targetlist=None):
        """ 
            
        t0range: [None or [tmin,tmax]]
            Should be a format automatically understood by astropy.time.Time
            e.g. t0range=["2018-04-01","2020-10-01"]
            
        """
        if clean_t0nan:
            data = self.data[~self.data["t0"].isna()]
        else:
            data = self.data.copy()
            
        # - Time Range
        if t0range is not None:
            t0_start = time.Time(t0range[0]).mjd
            t0_end = time.Time(t0range[1]).mjd
            data = data[data["t0"].between(t0_start, t0_end)]
        
        # - zorigin
        if z_quality is not None and z_quality not in ["any","all","*"]:
            data = data[data["z_quality"].isin(np.atleast_1d(z_quality))]
        
        if in_targetlist is not None:
            data = data.loc[np.asarray(in_targetlist)[np.in1d(in_targetlist, data.index.astype("string"))] ]
            
        if query:
            data = data.query(query)
            
        return data

    def get_goodlc_targets(self, n_early_points=">=2",
                               n_late_points=">=5",
                               n_points=">=10",
                               n_bands=">=2",
                               premax_range=[-15,0],
                               postmax_range=[0,30],
                               phase_range=[-15,30],
                               **kwargs):
        """ kwargs should have the same format as the n_early_point='>=2' for instance.
        None means no constrain, like n_bands=None means 'n_bands' is not considered.
        """
        query = {**dict(n_early_points=n_early_points, n_late_points=n_late_points,
                        n_points=n_points,n_bands=n_bands),
                 **kwargs}
        df_query = " and ".join([f"{k}{v}" for k,v in query.items() if v is not None])
        print(df_query)
        phase_coverage = self.get_phase_coverage(premax_range=premax_range,
                                                 postmax_range=postmax_range,
                                                 phase_range=phase_range)
        return phase_coverage.query(df_query).index.astype("string")

    def get_target_lightcurve(self, name):
        """ """
        from . import lightcurve
        return lightcurve.LightCurve.from_name(name)
    
    def get_phase_coverage(self,premax_range=[-15,0],
                                postmax_range=[0,30],
                                phase_range=[-15,30], min_det_perband=1):
        """ """        
        # All
        phases = self.phasedf[self.phasedf.between(*phase_range)].reset_index().rename({"level_0":"name"},axis=1)
        n_points = phases.groupby(["name"]).size().to_frame("n_points")
        n_bands = (phases.groupby(["name", "filter"]).size()>=min_det_perband
                        ).groupby(level=[0]).sum().to_frame("n_bands")
        # Pre-Max
        # - AnyBand
        premax = self.phasedf[self.phasedf.between(*premax_range)].reset_index().rename({"level_0":"name"},axis=1)
        n_early_points = premax.groupby(["name"]).size().to_frame("n_early_points")
        n_early_bands = (premax.groupby(["name", "filter"]).size()>=min_det_perband
                        ).groupby(level=[0]).sum().to_frame("n_early_bands")
        # - Per filters
        n_early_points_perfilter = premax.groupby(["name", "filter"]).size()
        n_early_points_g = n_early_points_perfilter.xs("p48g", level=1).to_frame("n_early_points_p48g")
        n_early_points_r = n_early_points_perfilter.xs("p48r", level=1).to_frame("n_early_points_p48r")
        n_early_points_i = n_early_points_perfilter.xs("p48i", level=1).to_frame("n_early_points_p48i")
        
        # Post-Max
        # - AnyBand        
        postmax = self.phasedf[self.phasedf.between(*postmax_range)].reset_index().rename({"level_0":"name"},axis=1)
        n_late_points = postmax.groupby(["name"]).size().to_frame("n_late_points")
        n_late_bands = (postmax.groupby(["name", "filter"]).size()>=min_det_perband
                       ).groupby(level=[0]).sum().to_frame("n_late_bands")
        # - Per filters
        n_late_points_perfilter = postmax.groupby(["name", "filter"]).size()
        n_late_points_g = n_late_points_perfilter.xs("p48g", level=1).to_frame("n_late_points_p48g")
        n_late_points_r = n_late_points_perfilter.xs("p48r", level=1).to_frame("n_late_points_p48r")
        n_late_points_i = n_late_points_perfilter.xs("p48i", level=1).to_frame("n_late_points_p48i")
            

        return pandas.concat([n_points,n_bands,
                              n_early_points,n_early_bands,n_late_points, n_late_bands,
                              n_early_points_g, n_early_points_r, n_early_points_i,
                              n_late_points_g, n_late_points_r, n_late_points_i], axis=1).fillna(0).astype(int)

        
    def build_phase_coverage(self, min_detection=5, groupby='filter', client=None, store=True, **kwargs):
        """ 
        time: 
        - Dask: 45s on a normal laptop | 4 cores 
        - No Dask: 60s on a normal laptop | 4 cores 
        """
        import pandas
        import dask
        from . import lightcurve

        phases = []
        datalc = self.get_data()
        datalc = datalc[datalc["redshift"].between(-0.1,0.2)]

        # - Without Dask
        if client is None:
            warnings.warn("loading without Dask (client is None) ; it will be slow")
            names_ok = []
            for name in datalc.index:
                try:
                    dt = lightcurve.LightCurve.from_name(name)
                    phases.append( dt.get_obsphase(min_detection=min_detection, groupby=groupby, **kwargs))
                    names_ok.append(name)
                except:
                    warnings.warn(f"get_obsphase did not work for {name}")
                    continue
                
            phasedf = pandas.concat(phases, keys=names_ok)

        # - With Dask
        else:
            for name in datalc.index:
                dt = dask.delayed(lightcurve.LightCurve.from_name)(name)
                phases.append( dt.get_obsphase(min_detection=min_detection, groupby=groupby, **kwargs)
                             )

            fphases = client.compute(phases)
            data_ = client.gather(fphases, "skip") # wait until all is done
            names = datalc.index[[i for i,f_ in enumerate(fphases) if f_.status=="finished"]]
            
            phasedf = pandas.concat(data_, keys=names)

        phasedf_exploded = phasedf.explode()
        if store:
            print("STORING DATA")
            filepath = io.get_phase_coverage(load=False)
            phasedf_exploded.to_csv(filepath)
            
        return phasedf_exploded
    
    
    # ------- #
    # PLOTTER #
    # ------- #

    
    def show_discoveryhist(self, ax=None, daymax=15, linecolor="C1", **kwargs):
        """ """
        from matplotlib.colors import to_rgba
        datasalt = self.get_data(clean_t0nan=True)
        
        if ax is None:
            fig = mpl.figure(figsize=[6,3])
            ax = fig.add_axes([0.1,0.2,0.8,0.7])
        else:
            fig = ax.figure
            

        prop = dict(fill=True, histtype="bar", 
                    density=False, facecolor=to_rgba("C0", 0.1), edgecolor="C0",
                   align="left", zorder=3)
        _ = ax.hist(datasalt.groupby("t0day").size(), range=[0,15], bins=15,
                    **{**prop,**kwargs})
        if linecolor != "None":
            ax.axvline(3.4, color="C1", zorder=5, lw=1.5, ls="--")
        
#        xx = np.arange(0, daymax)        
        #ax.scatter(xx[1:], stats.poisson.pmf(xx[1:], mu=3.5)*1050, 
        #           s=50, color="C1", zorder=5)

        #ax.set_yscale("log")

        _ = ax.set_xticks(np.arange(daymax))
        ax.set_xlim(0)

        clearwhich = ["left","right","top"] # "bottom"
        [ax.spines[which].set_visible(False) for which in clearwhich]
        ax.tick_params("y", labelsize="x-small")
        ax.set_yticks([0,100,200])

        ax.set_xlabel("Number of SN Ia per day")
        return fig
    
    
    def show_discoveryevol(self, ax=None, t0range=["2018-04-01","2021-01-01"],
                           typed_color="C0", quality_color="goldenrod",
                           xformat=True, dataprop={}, 
                           **kwargs):
        """ """
        from matplotlib.colors import to_rgba
        # - Data    
        datasalt_all = self.get_data(clean_t0nan=True, t0range=t0range, **dataprop)
        datasalt_lccut = self.get_data(clean_t0nan=True, t0range=t0range, 
                                        salt_quality=[3,7], **dataprop)    
        datasalt_zcut = self.get_data(clean_t0nan=True, t0range=t0range, 
                                        z_quality=2, **dataprop)    
        datasalt_zcut_lccut = self.get_data(clean_t0nan=True, t0range=t0range, 
                                        salt_quality=[3,7],
                                        z_quality=2, **dataprop)    


        # - Figue
        if ax is None:
            fig = mpl.figure(figsize=[8,4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure


        # - Internal

        def _show_single_(gb_cumsize, mplfunc="fill_between", 
                          add_text=True, text_color=None, **kwargs_):
            """ """
            time_ = time.Time(gb_cumsize.index, format="mjd").datetime
            values_ = gb_cumsize.values
            _ = getattr(ax,mplfunc)(time_, values_, **kwargs_)
            if add_text:
                ax.text(time_[-1], values_[-1],f" {values_[-1]}",
                            va="center",ha="left", color=text_color)

        # Show all
        _show_single_(datasalt_all.groupby("t0day").size().cumsum(), 
                     lw=2, ls="None", facecolor="w", edgecolor="None", 
                      add_text=False)
        _show_single_(datasalt_all.groupby("t0day").size().cumsum(), 
                     lw=2, ls="--", color=typed_color, mplfunc="plot",
                     text_color=typed_color)



        _show_single_(datasalt_lccut.groupby("t0day").size().cumsum(), 
                     lw=2, ls="None", facecolor=to_rgba(quality_color,0.2), 
                      edgecolor="None", add_text=False)
        _show_single_(datasalt_lccut.groupby("t0day").size().cumsum(), 
                     lw=2, ls="--", color=quality_color, mplfunc="plot",
                     text_color=quality_color)



        _show_single_(datasalt_zcut.groupby("t0day").size().cumsum(), 
                      mplfunc="plot",
                     lw=1, ls="-", color=typed_color, 
                     text_color=typed_color)

        _show_single_(datasalt_zcut_lccut.groupby("t0day").size().cumsum(), 
                     lw=2, ls="-", facecolor=to_rgba(quality_color,0.2), 
                      edgecolor="None", add_text=False)
        _show_single_(datasalt_zcut_lccut.groupby("t0day").size().cumsum(), 
                     lw=2, ls="-", color=quality_color, mplfunc="plot",
                     text_color=quality_color)


        # - Out Formating
        if xformat:
            from matplotlib import dates as mdates
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        clearwhich = ["left","right","top"] # "bottom"
        [ax.spines[which].set_visible(False) for which in clearwhich]

    #        ax.set_title("ZTF-1 Ia Statistics")
        ax.set_ylabel("Number of Type Ia Supernovae")
        ax.set_ylim(bottom=0)
        ax.tick_params("y", labelsize="small")
        return fig


    def _build_data_(self):
        """ """
        self._data= pandas.merge(self.salt2param, self.targetsdata, 
                                 left_index=True, right_index=True)
        
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def data(self):
        """ """
        if not hasattr(self,"_data") or self._data is None:
            self._build_data_()
        return self._data
    
    @property
    def salt2param(self):
        """ """
        return self._salt2param
        
    @property
    def targetsdata(self):
        """ """
        return self._targetsdata

    @property
    def phasedf(self):
        """ """
        if not hasattr(self, "_phasedf"):
            raise AttributeError("phase dataframe is jot yet loaded, see self.load_phasedf()")
        
        return self._phasedf
