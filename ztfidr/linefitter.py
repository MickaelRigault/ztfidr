import numpy as np

import pandas
from scipy import stats
from scipy.special import legendre

def get_truncnorm_prior(x, loc=0, scale=1, a=0, b=5):
    """ truncated normal prior.
        This truncation parameters (a and b) are in units of scale.
        such that parameters lower than: loc-a*scale and larger than loc+b*scale are set to 0
        """
    return stats.truncnorm.pdf(x, loc=loc, scale=scale, a=a, b=b)




class LineFitter( object ):
    LINES = {"HA":6562.8,
             "NII":6583.4,
            }
    JOINED_LINES = {"NII":{"lbda":6548.0, "amplitude":1/2.8}
                    }
    
    BACKGROUND_DEGREE = 3 # only affects guess, rest is self consistent.
    LBDA_WINDOM = 150
    def __init__(self, spectrum=None, redshift_guess=0):
        """ """
        self.spectrum = spectrum
        self.redshift_guess = redshift_guess

    # --------- #
    #   Data    #
    # --------- #
    def get_fitted_data(self, redshift_guess=None,  
                        check_variance=True, var_window_percent=10, normalised=True):
        """ """
        if redshift_guess is None:
            redshift_guess = self.redshift_guess
        
        lines = np.asarray(self.fitted_lines)
        lbda_flag = (self.spectrum.lbda > (lines.min()*(1+redshift_guess) - self.LBDA_WINDOM)) & \
                    (self.spectrum.lbda < (lines.max()*(1+redshift_guess) + self.LBDA_WINDOM))
        
        data = self.spectrum.data[lbda_flag].copy()
        data["lbda_x"] = ((data["lbda"].values - data["lbda"].min())/(data["lbda"].max() - data["lbda"].min()) - 0.5)*2
        if check_variance:
            print("checking variance")
            var_window = int(data.size*var_window_percent*0.5)
            var_ = np.mean([data["flux"][:var_window].values.std()**2,
                           data["flux"][-var_window:].values.std()**2])
            
            if  "variance" not in data:
                print("creating a variance")
                data["variance"] = var_
                
            else:
                var_in = np.mean([data["variance"][:var_window].values.mean(),
                                  data["variance"][-var_window:].values.mean()])
                print(f"variance correction, {var_/var_in}")
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

    def to_pandas(self):
        """ """
        results = self.fitted_parameters # dataframe of value, error, cov
        if self._fitdata_norm != 1: # that would also work, but useless
            results.loc[results.index.str.startswith("ampl_")] *= self._fitdata_norm
            results.loc[:,results.columns.str.startswith("cov_ampl_")] *=  self._fitdata_norm

        meta = pandas.Series({k:self._fit_output[k] for k in ["success","fun","nit","status", "message"]})
        return meta, results
        
    # --------- #
    #   Model   #
    # --------- #
    def fit(self, data=None, leastsq=False, use_redshift_guess=True, **kwargs):
        """ """
#        from iminuit import Minuit

    def fit_minuit(self, data=None, leastsq=False, use_redshift_guess=False,
                       limit_sigma=[2,10], limit_reshift=[0,0.3],
                       **kwargs):
        """ 

        Parameters
        ----------

        **kwargs goes to Minuit()
        """
        from iminuit import Minuit
        from iminuit.util import make_func_code
        setattr(self.__class__.get_logprob, "func_code", make_func_code(self.param_names))

        if data is not None:
            self._fitted_data = data

        # create the guesses            
        redshift = None if not use_redshift_guess else self.redshift_guess
        guess_list = self.get_parameterguess(redshift=redshift)
        guess = dict( zip(self.param_names, guess_list) )
        # load Minuit
        self._minuit = Minuit( self.get_logprob, **guess, **kwargs)
        self._minuit.limits["sigma"] = limit_sigma
        self._minuit.limits["redshift"] = limit_reshift
        # Fit
        self._minuit.migrad()
        self._minuit.hesse()
        return self.fitted_parameters

        
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
    
    def get_parameterguess(self, redshift=None, sigma=None,
                            data=None, **kwargs):
        """ """
        # Parameters are:
        # [amplitudes], redshift, sigma, [backgrounds] | [] means list
        if data is None:
            data = self.fitted_data

        # background
        b_params = np.zeros(self.BACKGROUND_DEGREE)
        b_params[0] = data["flux"].median() # constant. 
        if self.BACKGROUND_DEGREE>1:
            # devided by 2 because lbda_x defined between -1 and 1
            b_params[1] = (data["flux"].iloc[-5:].mean() - data["flux"].iloc[:5].mean())/2
            
        
        # Redshift
        top_line_args = data["flux"].argmax()
        if redshift is None:
            redshift = (data.iloc[top_line_args]["lbda"] / self.LINES["HA"]) - 1
            ha_arg = top_line_args
        else:
            ha_arg = np.argmin(np.abs(data["lbda"] - self.LINES["HA"]*(1+redshift)))
            
        if sigma is None:
            sigma = 4 # classique for lsf resolution
            
        ha_amplitude = (data.iloc[ha_arg]["flux"] - b_params[0]) * np.sqrt(2*np.pi)*sigma
        amplitudes = np.ones(self.nlines)*ha_amplitude 
        amplitudes[1:] /= 3 # all lines at a third of Ha
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
            redshift = self.redshift_guess
            _ = [ax.axvline(line*(1+redshift), ls="--", color="0.7")
                 for line in self.fitted_lines]
            
        if show_guess:
            parameters = self.get_parameterguess()
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
    def param_names(self):
        """ """
        return [f"ampl_{k.lower()}" for k in self.fitted_line_names] + \
               ["redshift", "sigma"] + \
               [f"ampl_back{k}" for k in range(self.BACKGROUND_DEGREE)]
    
