
import warnings
import numpy as np
from scipy import stats
import pandas
from . import io
from astropy import time, cosmology
_COEFS = 1.57
_COSMO = cosmology.Planck18
# Top level method #

def get_sample(**kwargs):
    """ Short to to Sample.load() """
    return Sample.load(**kwargs)


def get_data( saltmodel="default",
              good_coverage = True,
              good_lcfit = True,
              redshift_source=None,
              redshift_range=None,
              get_envprop = True, corr_mw=True,
                 **kwargs):
    """ Generic dataframe for the ZTF Cosmo DR2 sample.
   
    - good_coverage=True, means minimal sampling (see overview paper)
    - good_lcfit=True, means:
       - x1_range=[-3, 3],
       - c_range=[-0.2, 0.8],
       - t0_err_range=[0, 1],
       - x1_err_range=[0, 1], 
       - c_err_range=[0, 0.1],
       - fitprob > 1e-7

    use **kwargs to change any selection (see sample.get_data())
    """
    from .hostmass import get_taylor_hostmass
    sample = get_sample(saltmodel=saltmodel)
    
    prop = { **dict(good_coverage=good_coverage,
                    good_lcfit=good_lcfit,
                    redshift_range=redshift_range,
                    redshift_source =redshift_source),
              **kwargs}
    
    data = sample.get_data(**prop)
    
    # Hubble Residuals
    data['mb'] = -2.5 * np.log10( data["x0"] ) + 19*_COEFS
    data["mb_err"] =  +2.5/np.log(10) * data["x0_err"] / data["x0"]
    
    data['magres'] = data['mb'] - _COSMO.distmod(data["redshift"].values).value
    data["magres_err"] =  data["mb_err"].copy()
    data["cov_mag_c"] = -2.5*np.array(data['cov_x0_c'])/(np.log(10)*data['x0'])
    data["cov_mag_x1"] = -2.5*np.array(data['cov_x0_x1'])/(np.log(10)*data['x0'])
    
    # Environmental data
    if get_envprop:
        localmags = io.get_localhost_mag()
        globalmags = io.get_globalhost_mag().join(data[["redshift"]])

        if corr_mw:
            from .mw import correct_mwextinction_on_hostmag
            localmags = localmags.join(data["mwebv"])
            globalmags = globalmags.join(data["mwebv"])
            # apply the correction
            localmags = correct_mwextinction_on_hostmag(localmags)
            globalmags = correct_mwextinction_on_hostmag(globalmags)
            
        # get the host masses using Taylor's relation
        host_info = globalmags[["PS1i","PS1g","PS1i_err","PS1g_err", "redshift"]].dropna()
        mass, mass_err = get_taylor_hostmass(magi = host_info["PS1i"], magg = host_info["PS1g"],
                                         magi_err = host_info["PS1i_err"], magg_err = host_info["PS1g_err"],
                                         redshift=host_info["redshift"], use_prior=True)
        masses = mass.to_frame("mass").join(mass_err.to_frame("mass_err")).reindex(data.index)

        # get the color colors
        localmags["localcolor"] = localmags["PS1g"] - localmags["PS1z"]
        localmags["localcolor_err"] = np.sqrt(localmags["PS1g_err"]**2 + localmags["PS1z_err"]**2)

    
        #localmags["mass"] = localmags["PS1g"] - localmags["PS1z"]
        data = data.join(localmags[["localcolor","localcolor_err"]].loc[data.index])
        data = data.join(masses[["mass","mass_err"]].loc[data.index])
        data = data.join(globalmags[["PS1r"]].loc[data.index])
        
        ## Standardisation parameters
        lcolor_cut = 1
        data["h_lowcolor"] = stats.norm.cdf(lcolor_cut, loc=data["localcolor"], scale=data["localcolor_err"])
        data["h_lowcolor_err"] = 1e-3
    
        data["h_lowmass"] = stats.norm.cdf(10, loc=data["mass"], scale=data["mass_err"])
        data["h_lowmass_err"] = 1e-3

    return data.copy()


def _get_coverage_(phase_df, prefix=None, one_per_day=False):
    """ """
    if prefix is None:
        prefix = ""

    if one_per_day:
        phase_df = phase_df.copy().astype(int).reset_index().drop_duplicates().set_index(["ztfname","filter"])["phase"]
        
    n_points  = phase_df.groupby(level=0).size().to_frame(f"n_{prefix}points")
    n_bands   = ((phase_df.groupby(level=[0,1]).size()>0).groupby(level=0).sum()).to_frame(f"n_{prefix}bands")
    n_points_g = phase_df.xs("ztfg", level=1).groupby(level=0).size().to_frame(f"n_{prefix}points_ztfg")
    n_points_r = phase_df.xs("ztfr", level=1).groupby(level=0).size().to_frame(f"n_{prefix}points_ztfr")
    n_points_i = phase_df.xs("ztfi", level=1).groupby(level=0).size().to_frame(f"n_{prefix}points_ztfi")
    dfs = [n_points, n_bands, n_points_g, n_points_r, n_points_i]
    return pandas.concat(dfs, axis=1).fillna(0).astype( int )

class Sample():

    
    def __init__(self, data=None, saltmodel=None):
        """ """
        self.set_data(data)
        self._saltmodel = saltmodel
        
    @classmethod
    def load(cls, redshift_range=None, target_list=None, has_spectra=True,
                 saltmodel="default", **kwargs):
        """ Load a Sample instance building it from io.get_targets_data() """
        data, saltmodel = io.get_targets_data(saltmodel=saltmodel, **kwargs)
        
        if redshift_range is not None:
            data = data[data["redshift"].between(*redshift_range)]

        if target_list is not None:
            data = data.loc[target_list]

        if has_spectra:
            specfile = io.get_spectra_datafile(data=data)
            data = data[data.index.isin(specfile["ztfname"])]
            
        return cls(data=data, saltmodel=saltmodel)

    # ------- #
    # LOADER  #
    # ------- #
    def load_hostdata(self):
        """ load the host data using io.get_host_data(). This is made automatically upon hostdata call. """
        self.set_hostdata( io.get_host_data() )

    def load_spectra_df(self, add_phase=True, current_target=True):
        """ load the spectra dataframe 

        Parameters
        ----------
        add_phase: bool
            should the phase (based on spectra date and the target t0) be added

        current_target: bool
            only keep target that are in the current sample.
        
        Returns
        -------
        None
            sets self.spectra_df
        """
        spectra_df = io.get_spectra_datafile().set_index("ztfname")
        if current_target:
            spectra_df = spectra_df.loc[spectra_df.index.isin( self.data.index )]
            
        spectra_df["mjd"] = time.Time(np.asarray(spectra_df["date"].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}").astype(str).values, dtype="str")).mjd
        if add_phase:
            spectra_df = spectra_df.join( self.data["t0"] )
            spectra_df["phase"] = spectra_df["mjd"] - spectra_df["t0"]

        self._spectra_df = spectra_df
        
    def load_phase_df(self, min_detection=5, groupby='filter', client=None, rebuild=False, **kwargs):
        """ Load the phase_df. This is made automatically upton phase_df call.
        If this is the first time ever you call this, the phase_df has to be built, so it takes ~45s.
        Once built, the dataframe is stored such that, next time, it is directly loaded. 
        Use rebuild=True to bypass the automatic loading and rebuild the dataframe.
        
        Parameters
        ----------
        min_detection: [float] -optional-
            minimal signal to noise ratio for a point to be considered as 'detected'

        groupby: [string or list of] -optional-
            data column(s) to group the lightcurve data and measuring the count statistics.
            example: filter, [filter, fieldid]
             
        client: [dask.Client] -optional-
            if a dask.distributed.Client instance is given, this will use dask. 
            Otherwise a basic for loop (slower) is used.

        rebuild: [bool] -optional-
            force the dataframe to be rebuilt.

        **kwargs goes to self.build_phase_coverage()

        Returns
        -------
        None
        """
        if not rebuild:
            phase_df = io.get_phase_coverage(load=True, warn=False)
            if phase_df is None:
                rebuild = True
                
        if rebuild:
            phase_df = self.build_phase_coverage(min_detection=min_detection,
                                                    groupby=groupby,
                                                    client=client, store=True, **kwargs)
        self._phase_df = phase_df
        
    def load_fiedid(self):
        """ compute the fields containing the targets using 
        ztfquery.fields.get_fields_containing_target().
        This take quite some time.

        the 'fieldid' entry is added to self.data
        """
        from ztfquery import fields
        fieldid = [fields.get_fields_containing_target(s_["ra"],s_["dec"])
                       for i_,s_ in self.data.iterrows()]
        self.data["fieldid"] = fieldid

    # ------- #
    # SETTER  #
    # ------- #
    def set_data(self, data):
        """ attach to this instance the target data. 
        = Most likely you should not use this method directly. =
        use sample = Sample.load()
        """
        try:
            data["t0day"] = data["t0"].astype("int")
        except:
            data["t0day"] = data["t0"]
            
        self._data = data

    def set_hostdata(self, hostdata):
        """ attach to this instance the host data.
        = Most likely you should not use this method directly. =
        use sample.load_hostdata() or simply 
        sample.hostdata that automatically load this corrent hostdata.
        """
        self._hostdata = hostdata
        
    def merge_to_data(self, dataframe, how="outer", **kwargs):
        """ Merge the given dataframe with self.data. 
        The merged dataframe will replace self.data
        """
        self._data = pandas.merge(self.data, dataframe, how=how, 
                                 **{**dict(left_index=True, right_index=True),
                                 **kwargs})
        
    # ------- #
    # GETTER  #
    # ------- #
    def get_sncosmo_source(self):
        """ """
        from .lightcurve import get_saltmodel
        return get_saltmodel(self._saltmodel).source

    def get_data(self,
                     good_coverage=None,
                     good_lcfit=None,
                     clean_t0nan=True,                     
                     t0_range=None, x1_range=None, c_range=None,
                     redshift_range=None,
                     redshift_source=None,
                     t0_err_range=None, c_err_range=None, x1_err_range=None,
                     exclude_targets=None, in_targetlist=None,
                     ndetections=None,
                     fitprob=None,
                     classification=None,
                     first_spec_phase=None, query=None, data=None):
        """ 
        *_range: [None or [min, max]] -optional-
            cut to be applied to the data for
            t0, x1, c, (and there error) and redshift.
            boundaries are mandatory. For instance, redshift range lower than 0.06
            should be: redshift_range = (0, 0.06).
            = see below for t0 format =
        
        t0_range: [None or [tmin,tmax]] -optional-
            Should be a format automatically understood by astropy.time.Time
            e.g. t0_range=["2018-04-01","2020-10-01"]

        in_targetlist: [list of strings] -optional-
            The target must be in this list

        goodcoverage: [None or bool] -optional-
            Select the data given the lc phase coverage
            - None: no cut
            - True: only good lc kept
            - False: only bad lc kept (good lc discarded)
            This uses self.get_goodcoverage_targets(**coverage_prop)

        coverage_prop: [dict] -optional-
            kwargs passed to self.get_goodcoverage_targets
            = used only if goodcoverage is not None =

        query: [string] -optional-
            any additional query to be given to data.query({query}).
            This are SQL-like format applied to the colums. 
            See pandas.DataFrame.query()


        Returns
        -------
        DataFrame (sub part or copy of self.data)
        """
            
        if clean_t0nan:
            data = self.data[~self.data["t0"].isna()]
        else:
            data = self.data.copy()
        
        # - Time Range
        if t0_range is not None:
            t0_start = time.Time(t0_range[0]).mjd
            t0_end = time.Time(t0_range[1]).mjd
            data = data[data["t0"].between(t0_start, t0_end)]

        # LC Cuts
        # - stretch (x1) range
        if x1_range is not None:
            data = data[data["x1"].between(*x1_range)]

        # - color (c) range
        if c_range is not None:
            data = data[data["c"].between(*c_range)]

        # - t0 errors  range
        if t0_err_range is not None:
            data = data[data["t0_err"].between(*t0_err_range)]
            
        # - stretch errors (x1) range
        if x1_err_range is not None:
            data = data[data["x1_err"].between(*x1_err_range)]

        # - color errors (c) range
        if c_err_range is not None:
            data = data[data["c_err"].between(*c_err_range)]


        # Redshift Cuts
        # - Redshift range
        if redshift_range is not None:
            data = data[data["redshift"].between(*redshift_range)]
            
        # -  redshift origin
        if redshift_source is not None:
            data = data[data["source"].isin(np.atleast_1d(redshift_source))]


        # Classification
        if classification is not None:
            data = data[data["sn_type"].isin(np.atleast_1d(classification))]
            
        # Target cuts
        # - in given list
        if in_targetlist is not None:
            data = data.loc[np.asarray(in_targetlist)[np.in1d(in_targetlist, data.index.astype("string"))] ]

        if exclude_targets is not None:
            data = data.loc[~data.index.isin(exclude_targets)]


        # LC Coverage
        if ndetections is not None:
            phase_coverage = self.get_phase_coverage()["n_points"]
            min_corevarge = phase_coverage[phase_coverage>=ndetections]
            data = data.loc[data.index.isin(min_corevarge.index)]
            
        # - special good lc list.            
        if good_coverage is not None:
            if good_coverage:
                data = data[data["lccoverage_flag"].astype(bool)]
            else:
                data = data[~data["lccoverage_flag"].astype(bool)]


        if good_lcfit is not None:
            if good_lcfit:
                data = data[data["fitquality_flag"].astype(bool)]
            else: # lcquality_flag
                data = data[~data["fitquality_flag"].astype(bool)]
                
        if fitprob is not None:
            data = data[data["fitprob"]>fitprob]
                
        # Spectral Cut.
        if first_spec_phase is not None:
            first_spec = self.spectra_df.groupby(level=0).phase.first()
            first_spec = first_spec[first_spec<=first_spec_phase]
            data = data.loc[data.index.isin(first_spec.index)]

        # Additional Query
        if query:
            data = data.query(query)


        bitmask = np.zeros(len(data))
        for i, (value, range) in enumerate( [["x1", [-3,3]], #2**0
                                     ["c", [-0.2, 0.8]],#2**1
                                     ["t0_err", [0, 1]],#2**2
                                     ["x1_err", [0, 1]],#2**3
                                     ["c_err", [0, 0.1]],#2**4
                                     ["fitprob", [1e-7, 10]]]):#2**5
            flag_in = data[value].between(*range).values
            bitmask[~flag_in] += 2**i
        data["fitquality_bitmask"] = np.asarray(bitmask, dtype=int)
    
        return data

    def get_phases(self, one_per_day=False,
                         phase_range=None,
                         detection=5, incl_errorfloor=True,
                         in_targetlist=None,
                         flagout=[1, 2, 4, 8, 16],
                         restframe=True):
        """ """
        phase_df = self.phase_df.copy()
        
        if restframe:
            phase_df = phase_df.join(self.data["redshift"])
            phase_df["phase"] = phase_df["phase"]/(1+phase_df["redshift"])

        if phase_range is not None:
            phase_df = phase_df[phase_df["phase"].between(*phase_range)]
        
        # remove bad lc points
        if flagout is not None:
            flag_ = np.all([(phase_df["flag"]&i_==0) for i_ in np.atleast_1d(flagout)], axis=0)
            phase_df = phase_df[flag_]


        if detection is not None:
            if incl_errorfloor:
                det_key = "detection_ef"
            else:
                det_key = "detection"
            phase_df = phase_df[(phase_df[det_key]>=detection)]["phase"].copy()
        else:
            phase_df = phase_df["phase"]

        if one_per_day:
            phase_df = phase_df.astype(int).reset_index().drop_duplicates().set_index(["ztfname","filter"])["phase"]
            
        if in_targetlist:
            phase_df = phase_df[phase_df.index.get_level_values(0).isin(np.atleast_1d(in_targetlist))]
            
        return phase_df
                                    
    
    # Target | High level
    def get_target(self, name):
        """ """
        from . import target
        lightcurve = self.get_target_lightcurve(name)
        spectra = self.get_target_spectra(name, as_spectra=False)
        meta = self.data.loc[name]
        return target.Target(lightcurve, spectra, meta=meta)

    def get_target_typing(self, name=None):
        """ """
        if name is None:
            return self.data["sn_type"].copy()
        
        return self.data.loc[np.atleast_1d(name)]["sn_type"].copy()

    # model
    def get_target_saltmodel(self, name):
        """ """
        from .lightcurve import get_saltmodel
        propmodel = self.data.loc[name].rename({"redshift":"z"})[["z","t0","x0","x1","c","mwebv"]].to_dict()
        return get_saltmodel(which=self._saltmodel, **propmodel)
        
    # LightCurve
    def get_target_lightcurve(self, name, **kwargs):
        """ Get the {name} LightCurve object """
        from . import lightcurve
        return lightcurve.LightCurve.from_name(name, saltparam=self.data.loc[name], saltmodel=self._saltmodel)

    def get_target_lightcurve_data(self, name,  phase_range=None, mjd_range=None, **kwargs):
        """ Get the {name} LightCurve object """
        from . import lightcurve
        return lightcurve.get_target_lcdata(name, saltparam=self.data.loc[name],  phase_range=None, mjd_range=None, **kwargs)
    
    def get_target_lightcurve_residual(self, name, phase_range=None, restphase=False, mjd_range=None, **kwargs):
        """ fet the {name} lightcurve residuals given the target's salt model. 
        
        Parameters
        ----------
        name: str
            target's name

        phase_range: [float, float]
            limit phase range (days to maximum of light t0): [start, end]
        
        **kwargs goes to lightcurve.LightCurve.get_model_residual

        Returns
        -------
        pandas.DataFrame
        """
        from . import lightcurve
        if phase_range is not None and restphase:
            phase_range = np.asarray(phase_range)/(1+self.data.loc[name]["redshift"])
        
        return lightcurve.get_target_lcresiduals(name, phase_range,
                                                     mjd_range=mjd_range,
                                                     saltparam=self.data.loc[name],
                                                     which_model=self._saltmodel,
                                                     **kwargs)
    
    # Spectrum
    def get_target_spectra(self, name, **kwargs):
        """ Get a list with all the Spectra for the given object """
        from . import spectroscopy
        return spectroscopy.Spectrum.from_name(name, **kwargs)

    # Extra
    def get_goodcoverage_targets(self, detection=5, one_per_day=True,
                                      **kwargs):
        """ kwargs should have the same format as the n_early_point='>=2' for instance.
        None means no constrain, like n_bands=None means 'n_bands' is not considered.
        """
        # Only considering 1 point per band and per day (mjd int) if one_per_day==True
        query = {**dict(n_points=">=7", # any band
                        n_early_points=">=2", # e.g. at least 2g or 1g and 1i
                        n_late_points=">=2", # e.g. at least 2g or 1g and 1i
                        n_bands=">=2", # e.g. at least 1g and 1r | 2 filters post max
                        ),
                 **kwargs}
        df_query = " and ".join([f"{k}{v}" for k,v in query.items() if v is not None])

        phase_coverage = self.get_phase_coverage(detection=detection,
                                                 one_per_day=one_per_day)
        
        return phase_coverage.query(df_query).index.astype("string")
    
    def get_phase_coverage(self, premax_range = [-10, 0],
                                 postmax_range = [0, 40],
                                 phase_range = [-10, 40],
                                 one_per_day=True,
                                 detection=5):
        """ """
        # force phase range first then bin per day | don't been yet
        phase_df = self.get_phases(one_per_day=False,
                                   detection=detection)

        dfs = _get_coverage_( phase_df[phase_df.between(*phase_range)],
                                  one_per_day=one_per_day
                            )
        
        dfs_early = _get_coverage_(phase_df[phase_df.between(*premax_range)],
                                       prefix="early_", one_per_day=one_per_day)
        dfs_late = _get_coverage_(phase_df[phase_df.between(*postmax_range)],
                                      prefix="late_", one_per_day=one_per_day)
        
        return pandas.concat([dfs, dfs_early, dfs_late], axis=1
                            ).reindex(self.data.index).fillna(0).astype(int)
        
    def build_phase_coverage(self, groupby='filter', client=None, store=True,
                                 error_floor={"ztfg":2.5, "ztfr":3.5, "ztfi":6},
                                 **kwargs):
        """ 
        time: 
        - Dask: 45s on a normal laptop | 4 cores 
        - No Dask: 60s on a normal laptop | 4 cores 
        """
        warnings.warn("building phase coverage takes ~30s.")
        
        import pandas
        from . import io
        phases = []
        data = self.get_data()
        
        lcdata = io.get_lightcurve_datafile(contains=".csv").set_index("ztfname").loc[data.index]
        dfs = pandas.concat([pandas.read_csv(f_, delim_whitespace=True, comment='#') for f_ in lcdata["fullpath"]],
                                keys=data.index)
        dfs["phase"] = dfs["mjd"] - data["t0"].reindex(dfs.index, level=0)
        dfs["detection"] = dfs["flux"]/dfs["flux_err"]
        # error floor:
        error_floor=pandas.Series(error_floor, name="errorfloor").to_frame().reset_index().rename({"index":"filter"},
                                                                                                   axis=1)
        dfs = dfs.reset_index().merge(error_floor, on="filter")
        dfs["detection_ef"] = dfs["flux"]/ (np.sqrt(dfs["flux_err"]**2 + (dfs["flux"]*dfs["errorfloor"]/100)**2) )
        phase_df = dfs.reset_index().set_index(["ztfname","filter"])[["phase","detection",
                                                                          "detection_ef",
                                                                      "flag","in_baseline"]]
        if store:
            filepath = io.get_phase_coverage(load=False)
            phase_df.to_parquet(filepath)
            
        return phase_df
    
    
    # ------- #
    # PLOTTER #
    # ------- #
    def show_ideogram(self, key, keyerr=None,
                          ax=None, data=None, dataprop={},
                          facecolor=None, edgecolor=None, lw=2,
                          nsigma_range=3, npoints=1000,
                          density=False, normed_one=False,
                          **kwargs):
        """ """
        from matplotlib.colors import to_rgba

        # axes
        if ax is None:
            import matplotlib.pyplot as mpl
            fig = mpl.figure(figsize=[6,3])
            ax = fig.add_axes([0.1,0.2,0.8,0.7])
        else:
            fig = ax.figure

        # data            
        if keyerr is None:
            keyerr = key+"_err"

        if data is None:
            data = self.get_data(**dataprop)

        # getting the x and y            
        d, d_err = data[[key, keyerr]].values.T
        d_range = np.nanmin(d-nsigma_range*d_err), np.nanmax(d+nsigma_range*d_err)
        dd = np.linspace(*d_range, npoints)
        dideo = np.sum(stats.norm.pdf(dd[:,None],
                                      loc=d, scale=d_err), 
                           axis=1)
        
        # plotting properties
        if facecolor is None:
            facecolor = to_rgba("C0",0.3)
            facecolor_ = "C0"
        else:
            facecolor_ = facecolor
            
        if edgecolor is None:
            edgecolor = to_rgba(facecolor_, 1)


        if density:
            dideo /= np.nansum(dideo)
            
        if normed_one:
            dideo /= np.nanmax(dideo)
            
        # plot
        _ = ax.fill_between(dd, dideo, 
                                facecolor=facecolor, 
                                edgecolor=edgecolor, lw=lw,
                                **kwargs)
        
        # fancy
        ax.set_ylim(0)
        return fig
    
    def show_discoveryhist(self, ax=None, daymax=15, linecolor="C1", **kwargs):
        """ """
        from matplotlib.colors import to_rgba
        datasalt = self.get_data(clean_t0nan=True)
        
        if ax is None:
            import matplotlib.pyplot as mpl
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
    
    
    def show_discoveryevol(self, ax=None, t0_range=["2018-04-01","2021-01-01"],
                           typed_color="C0", quality_color="goldenrod",
                           xformat=True, dataprop={}, 
                           **kwargs):
        """ """
        from matplotlib.colors import to_rgba

        # - Data    
        datasalt_all = self.get_data(clean_t0nan=True, 
                                        t0_range=t0_range,
                                        **dataprop)
        
        datasalt_lccut = self.get_data(clean_t0nan=True,
                                        t0_range=t0_range,
                                        goodcoverage=True,
                                        **dataprop)
        
        datasalt_zcut = self.get_data(clean_t0nan=True,
                                        t0_range=t0_range, 
                                        z_quality=2,
                                        **dataprop)
        
        datasalt_zcut_lccut = self.get_data(clean_t0nan=True,
                                        t0_range=t0_range, 
                                        z_quality=2,
                                        goodcoverage=True,
                                        **dataprop)

        
        # - Figue
        if ax is None:
            import matplotlib.pyplot as mpl            
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



        
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def data(self):
        """ """
        return self._data

    @property
    def hostdata(self):
        """ """
        if not hasattr(self, "_hostdata"):
            self.load_hostdata()
        return self._hostdata
    
    @property
    def phase_df(self):
        """ """
        if not hasattr(self, "_phase_df"):
            self.load_phase_df()
        
        return self._phase_df

    @property
    def phase_coverage(self):
        """ """
        if not hasattr(self, "_phasecoverage"):
            self._phasecoverage = self.get_phase_coverage() # default
            
        return self._phasecoverage
    
    @property
    def spectra_df(self):
        """ dataframe containing the spectral information """
        if not hasattr(self, "_spectra_df"):
            self.load_spectra_df()
        
        return self._spectra_df
