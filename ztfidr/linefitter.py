import numpy as np

import pandas
from scipy import stats
from scipy.special import legendre


__all__ = ["fit_spectrum"]

def fit_spectrum(spectrum, redshift, figure=True, dirout=None, lbda_window=None, **kwargs):
    """ loads the intance and fit a spectrum 
        
    Parameters
    ----------
    spectrum: ztfidr.Spectrum or str
        spectrum or spectrum filepath to fit

    redshift: float
        initial redshift guess

    figure: bool
        should a figure be created. 
        If False, returned figure is None.

    dirout: str
        path for the directory where results will be stored. 
        if figure is True, the figure will also be stored.
        
    **kwargs fit() options
        
    Returns
    -------
    pandas.Series, pandas.DataFrame
        metadata table and results (value, error, cov_)
    """
    return LineFitter.fit_spectrum(spectrum, redshift, figure=figure, dirout=dirout,
                                       lbda_window=lbda_window, **kwargs)

class LineFitter( object ):
    LINES = {"HA":6562.8,
             "NII":6583.4,
            }
    JOINED_LINES = {"NII":{"lbda":6548.0, "amplitude":1/2.8}
                    }
    
    BACKGROUND_DEGREE = 3 # only affects guess, rest is self consistent.
    LBDA_WINDOW = 150  # value used it not provided.
    
    def __init__(self, spectrum=None, redshift_init=0, lbda_window=None):
        """ """
        self.spectrum = spectrum
        self.redshift_init = redshift_init
        
        if lbda_window is None:
            lbda_window = self.LBDA_WINDOW
        
        self.lbda_window = lbda_window

    @classmethod
    def fit_spectrum(cls, spectrum, redshift, figure=True, dirout=None,
                         lbda_window=None, **kwargs):
        """ loads the intance and fit a spectrum 
        
        Parameters
        ----------
        spectrum: ztfidr.Spectrum or str
            spectrum or spectrum filepath to fit

        redshift: float
            initial redshift guess
            
        figure: bool
            should a figure be created. 
            If False, returned figure is None.

        **kwargs fit() options
        
        Returns
        -------
        (pandas.Series, pandas.DataFrame), figure
            metadata table and results (value, error, cov_)
        """
        if type(spectrum) is str:
            from .spectroscopy import Spectrum
            spectrum = Spectrum.from_filename(spectrum)
        # Loads
        this = cls(spectrum, redshift, lbda_window=lbda_window)
        # Fit        
        params = this.fit(**kwargs)
        # Plot
        if figure:
            fig = this.show_data(show_guess=True, 
                                    model_parameters=params["value"])
        else:
            fig = None
        # Store
        if dirout is not None:
            _ = this.store(dirout, fig=fig)
            
        return *this.to_pandas(), fig

    # ---------- #
    # High Level #
    # ---------- #
    def fit(self, guess={}, limits={}, fixed={},
                limit_sigma=[.01, 5], limit_reshift=[0,0.4],
                sigma_pixel=True,
                **kwargs):
        """ fit the model. 

        The fitted data are these from self.fitted_data that is created
        automatically up on first call using get_fitted_data(). 
        
        Parameters
        ----------
        guess: dict
            dictionary to overwrite any default guess parameters
            obtained using self.get_guess_parameters(). 

        limit_sigma: (float, float)
            boundaries of the sigma parameters (emission line 'scale' in lbda-pixel)

        limit_redshift: (float, float)
            boundaries for the fitted redshift.
    
        **kwargs goes to Minuit()

        Returns
        -------
        DataFrame

        See also
        --------
        get_fitted_data: get the data as fitted (truncated as needed and normalize)
        to_pandas: get the fit result in pandas format (meta, result)
        """
        if fixed is None:
            fixed = {}
            
        if limits is None:
            limits = {}
        
        # red input
        if limit_sigma is not None and sigma_pixel:
            average_step = self.fitted_data["lbda"].diff().mean()
            limit_sigma = list(np.asarray(limit_sigma) * average_step)
            sigma = np.sqrt(2**2 + average_step**2) # 2 A corresponds to ~90 km/s gas dispersion
        else:
            sigma = None

        
        from iminuit import Minuit
        from iminuit.util import make_func_code
        # Minuit naming
        setattr(self.__class__.get_logprob, "func_code", make_func_code(self.param_names))

        # create the guesses
        guess_list = self.get_guess_parameters(sigma=sigma)
        guess = {**dict( zip(self.param_names, guess_list) ), **guess}
        guess = {**guess, **fixed} # force the fixed parameters
        self._fit_guess = guess

        #
        # load Minuit
        #
        # -> function to optimize
        self._minuit = Minuit( self.get_logprob, **guess, **kwargs)
        
        # -> parameter limits
        self._minuit.limits["sigma"] = limit_sigma
        self._minuit.limits["redshift"] = limit_reshift
        for k, v in limits.items():
            self._minuit[k] = v

        # -> parameter to fix
        for k, v in fixed.items(): # value set in the guess.
            self._minuit.fixed[k] = True

        # 
        # Run the fit
        #
        self._minuit.migrad()
        self._minuit.hesse()
        
        # output 
        return self.fitted_parameters

    def store(self, dirout, fig=None):
        """ """
        import os
        filename = os.path.basename(self.spectrum.filename).replace('.ascii', f'_linfit.parquet')
        filepath = os.path.join(dirout, filename)
        meta, results = self.to_pandas()
        results.to_parquet(filepath)
        if fig is not None:
            fig.savefig( filepath.replace(".parquet",".pdf") )

        return filepath
    
    def to_pandas(self, restore_norm=True):
        """ convert the minuit output into pandas format

        Parameters
        ----------
        restore_norm: bool
            the fitted data are normalized. Should the returned amplitude parameters
            be restoted into input data units ? 
            If False, returned results will be in fitted_data units ; 
            as for self.fitted_parameters.

        Returns
        -------
        pandas.Series, pandas.DataFrame
            metadata table and results (value, error, cov_)

        See also
        --------
        fitted_parameters: fit results in pandas dataframe
        fit: main fit method.
        """
        results = self.fitted_parameters # dataframe of value, error, cov
        results = results.join( pandas.Series(self._fit_guess, name="guess"))

        meta = pandas.Series({k: getattr(self._minuit,k) for k in ["fval","valid","accurate","nfit","nfcn"]})
        meta["ha_detection"] = (results["value"]/results["error"]).loc["ampl_ha"]
        meta["dof"] = len(self.fitted_data) - len(self.free_parameters)
        meta["targetname"] = self.spectrum.targetname
        
        if self._fitdata_norm != 1 and restore_norm: # that would also work, but useless
            results.loc[results.index.str.startswith("ampl_")] *= self._fitdata_norm
            results.loc[:,results.columns.str.startswith("cov_ampl_")] *=  self._fitdata_norm


        return meta, results

    # --------- #
    #  Method   #
    # --------- #
    def get_fitted_data(self, redshift_init=None,  
                        check_variance=True, var_window_percent=10,
                        normalised=True):
        """ get a subpart of the data where to perform the fit. 
        They are truncated in wavelength to the relevant lbda range and normalized. 

        Parameters
        ----------
        redshift_init: float
            initial redshift guess used to truncate the wavelength range of interest
            (see self.lbda_window parameters).
            If None self.redshift_init is used (recommended)

        check_variance: bool
            should this method fix any potential variance issues. 
            Under or over estimated variance affects the fitted parameter 
            error estimation. Scatter of the data at the fitted_data edges 
            are used for that. (see var_window_percent).
            This also created a variance estimation if no variance exist.

        var_window_percent: float
            = ignored if check_variance is False = 
            the pixel-window (in percent of total fitted_data lbda) used 
            to estimate the expected signal variance ; half from each edges 
            of the fitted_data wavelength range. 
            For instance, var_window_percent=10 means that 10% of the 
            spectrum will be used to estimate the variance (flux std) 
            using the average of the first and last 5%.
            
        normalised: bool
            should the data be normalized (flux.mean) for the fit ?
            (highly recommended)
        
        Returns
        -------
        pandas.DataFrame
            lbda, flux[, variance]
           
        """
        if redshift_init is None:
            redshift_init = self.redshift_init
        
        lines = np.asarray(self.fitted_lines)
        lbda_flag = (self.spectrum.lbda > (lines.min()*(1+redshift_init) - self.lbda_window)) & \
                    (self.spectrum.lbda < (lines.max()*(1+redshift_init) + self.lbda_window))
        
        data = self.spectrum.data[lbda_flag].copy()
        data["lbda_x"] = ((data["lbda"].values - data["lbda"].min())/(data["lbda"].max() - data["lbda"].min()) - 0.5)*2
        if check_variance:
            var_window = int(data.size*var_window_percent/100*0.5)
            var_ = np.mean([data["flux"][:var_window].values.std()**2,
                           data["flux"][-var_window:].values.std()**2])
            
            if  "variance" not in data:
                data["variance"] = var_
                
            else:
                var_in = np.mean([data["variance"][:var_window].values.mean(),
                                  data["variance"][-var_window:].values.mean()])
                data["variance"] *= (var_/var_in)

        if normalised:
            norm = np.nanmean(data["flux"])
            data["flux"] /= norm
            if "variance" in data:
                data["variance"] /= norm**2
            self._fitdata_norm = norm
        else:
            self._fitdata_norm = 1

        return data
        
    # --------- #
    #   Model   #
    # --------- #
    def get_logprob(self, *parameters):
        """ """
        chi2 = self.get_chi2(parameters)
        priors = 0 # to be addid if needed.
        return chi2 + priors
        
    def get_chi2(self, parameters, leastsq=False, data=None):
        """ """
        data = self.get_model(parameters, data=data)
        to_minimize = data["residual"].values**2
        if not leastsq and "variance" in data:
            to_minimize /= data["variance"].values
            
        return np.nansum(to_minimize)
        
    def parse_parameters(self, parameters):
        """ """
        amplitudes = parameters[:self.nlines]
        redshift, sigma_lines, *b_parameters = parameters[self.nlines:]
        return list(amplitudes), redshift, sigma_lines, list(b_parameters)

    
    def get_linemodel(self, lbda, amplitudes, redshift, sigma):
        """ """
        all_amplitudes = np.asarray(self.get_lineamplitudes(amplitudes))
        all_lbda = np.asarray(self.fitted_lines) * (1 + redshift)

        all_model = np.dot(all_amplitudes,
                           stats.norm.pdf(np.asarray(lbda)[:,None], 
                                          loc=all_lbda, 
                                          scale=sigma).T)
        return all_model
    
    def get_lineamplitudes(self, amplitudes):
        """ build full amplitudes, adding given amplitudes to joined ones. """
        dict_ampl = dict(zip(self.fitted_line_names, amplitudes))
        joined_ampl = [self.JOINED_LINES.get(k).get("amplitude", np.NaN)*dict_ampl[k] 
                      for k in self.fitted_line_names if k in self.JOINED_LINES]
        return amplitudes + joined_ampl
    
    @staticmethod
    def get_backgroundmodel(parameters, lbda_x):
        """ """
        background = np.dot(parameters, np.stack([legendre(id_)(lbda_x) 
                                               for id_ in range( len(parameters) )])
                           )
        return background
    
    def get_model(self, parameters, data=None, verbose=False, **kwargs):
        """ """
        if data is None:
            data = self.fitted_data
            
        # Parse inputs
        amplitudes, redshift, sigma_lines, b_parameters = self.parse_parameters(parameters)
        if verbose:
            print(f"amplitudes: {amplitudes}")
            print(f"redshift: {redshift}")
            print(f"sigma_lines: {sigma_lines}")
            print(f"b_parameters: {b_parameters}")            
        
        
        # background
        background = self.get_backgroundmodel(b_parameters, data["lbda_x"])
        # lines
        lines = self.get_linemodel(data["lbda"], amplitudes, redshift, sigma_lines)
        # add it to the dataframe
        data["model"] = background + lines
        data["residual"] = data["model"] - data["flux"]
        
        return data
    
    def get_guess_parameters(self, sigma=None, redshift_buffer=10, **kwargs):
        """ 

        """
        #
        # background
        #
        b_params = np.zeros(self.BACKGROUND_DEGREE)
        b_params[0] = self.fitted_data["flux"].median() # constant. 
        if self.BACKGROUND_DEGREE>1:
            # devided by 2 because lbda_x defined between -1 and 1
            b_params[1] = (self.fitted_data["flux"].iloc[-5:].mean() - self.fitted_data["flux"].iloc[:5].mean())/2
            
        #
        # redshift
        #
        top_line_args = self.fitted_data["flux"].argmax()
                
        if redshift_buffer is not None:
            ha_arg = np.argmin(np.abs(self.fitted_data["lbda"] - self.LINES["HA"]*(1+self.redshift_init)))
            # 4 pixels buffer
            buffer_size = 2
            local_argmax = self.fitted_data.iloc[ha_arg-buffer_size : ha_arg+buffer_size]["flux"].argmax()
            top_line_args = ha_arg - buffer_size + local_argmax

        redshift = (self.fitted_data.iloc[top_line_args]["lbda"] / self.LINES["HA"]) - 1

        #
        # sigma
        #
        if sigma is None:
            average_step = self.fitted_data["lbda"].diff().mean()
            sigma = np.sqrt(2**2 + average_step**2) # 2 A corresponds to ~90 km/s gas dispersion
        # - Nothing do do
        
        #
        # Amplitudes
        # 
        ha_amplitude = (self.fitted_data.iloc[top_line_args]["flux"] - b_params[0]) * np.sqrt(2*np.pi)*sigma
        amplitudes = np.ones(self.nlines)*ha_amplitude 
        amplitudes[1:] /= 3 # all lines at a third of Ha
        # returns
        return *amplitudes, redshift, sigma, *b_params
        
        
    def show_data(self, ax=None, show_lines=True, show_error=True, 
                 show_guess=False, model_parameters=None, 
                  legend=True):
        """ """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=[7,5])
        ax = fig.add_subplot(111)
        
        data = self.get_fitted_data().astype(float) # does a copy
        ax.plot(data.lbda, data.flux, label="data")
        ax.set_xlabel("wavelength [A]", fontsize="large")
        ax.set_ylabel("flux []", fontsize="large")
        
        
        if "variance" in data and show_error:
            err = np.sqrt(data["variance"])
            ax.fill_between(data.lbda, data.flux+err, data.flux-err, alpha=0.1)
        
        if show_lines:
            redshift = self.redshift_init
            _ = [ax.axvline(line*(1+redshift), ls="--", color="0.7")
                 for line in self.fitted_lines]
            
        if show_guess:
            if hasattr(self, "_fit_guess"):
                parameters = [self._fit_guess[k] for k in self.param_names]
            else:
                parameters = self.get_guess_parameters()
                
            data = self.get_model(parameters, data=data)
            ax.plot(data["lbda"], data["model"], ls=":", label="guess")
            
        if model_parameters is not None:
            data = self.get_model(model_parameters, data=data)
            ax.plot(data["lbda"], data["model"], ls="-", label="model")
            
        if legend:
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=3, mode="expand", borderaxespad=0.)
        return fig
        
    # ============ #
    #  Properties  #
    # ============ #
    @property
    def nlines(self):
        """ """
        return len(self.LINES)
    
    @property
    def fitted_line_names(self):
        """ """
        return list(self.LINES.keys())

    @property
    def fitted_data(self):
        """ """
        if not hasattr(self, "_fitted_data") or self._fitted_data is None:
            self._fitted_data = self.get_fitted_data()
            
        return self._fitted_data
    
    @property
    def fitted_lines(self):
        """ """
        base_lines = [self.LINES.get(k, np.NaN) for k in self.fitted_line_names]
        joined_lines = [self.JOINED_LINES.get(k).get("lbda", np.NaN) 
                        for k in self.fitted_line_names if k in self.JOINED_LINES]
        return base_lines+joined_lines
    
    @property
    def fitted_parameters(self):
        """ """
        dataout = pandas.DataFrame(self._minuit.values, columns=["value"], index=self._minuit.parameters)
        dataout["error"] = np.sqrt(np.diag(self._minuit.covariance))
        cov = pandas.DataFrame( self._minuit.covariance, 
                               index=dataout.index, columns="cov_"+dataout.index)
        
        results = dataout.join(cov)
        return results

    @property
    def free_parameters(self):
        """ """
        if not hasattr(self,"_minuit"):
            raise AttributeError("Minuit has not run yet.")
        
        return [k for k,v in self._minuit.fixed.to_dict().items() if not v]
    
    @property
    def param_names(self):
        """ """
        return [f"ampl_{k.lower()}" for k in self.fitted_line_names] + \
               ["redshift", "sigma"] + \
               [f"ampl_back{k}" for k in range(self.BACKGROUND_DEGREE)]
    
