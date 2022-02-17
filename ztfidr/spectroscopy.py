import numpy as np
import os
import pandas
import warnings
from . import io

SPEC_DATAFILE = io.get_spectra_datafile()


def read_spectrum(file_):
    """ """
    data = open(file_).read().splitlines()
    try:
        header = pandas.DataFrame([d.replace("#","").replace(":"," ").replace("=","").split()[:2] for d in data 
                                   if (d.startswith("#") or "=" in d or ":" in d) and len(d.split())>=2],
                              columns=["keys", "values"]).set_index("keys")["values"]
    except:
        raise IOError(f"header build Fails for {file_}")
    try:
        lbda, flux, *variance = np.asarray([d.split() for d in data 
                                        if not (d.startswith("#") 
                                                or d.startswith("COMMENT") or d.startswith("HISTORY")
                                                or "=" in d or ":" in d) and len(d.split())>=2]).T
        data = pandas.DataFrame({"lbda":lbda, "flux":flux}, dtype="float")
    except:
        raise IOError(f"dataframe build Fails for {file_}")
    if len(variance)>0:
        data["variance"] = variance[0]
    return header, data


class Spectrum( object ):
    """ """
    def __init__(self, data, header=None, meta=None, use_dask=False, snidresult=None,
                     filename=None, load_snidres=True):
        """ """
        self.set_data(data)
        self.set_header(header)
        self.set_meta(meta)
        self.set_snidresult(snidresult)
        # - Dask
        self._use_dask = use_dask
        # - Filename        
        self._filename = filename
        # - snid result
        if load_snidres:
            self.load_snidresult(allow_fit=False)

        
    @classmethod
    def from_filename(cls, filename, use_dask=False, snidresult=None,
                          load_snidres=True, **kwargs):
        """ 
        load_snidres: fetch for snidresults and loads it if it exists.
        """
        if not use_dask:
            header, data = read_spectrum(filename)
        else:
            from dask import delayed
            header_data = delayed(read_spectrum)(filename)
            header = header_data[0]
            data = header_data[1]
            
        meta = io.parse_filename(filename)
        return cls(data, header=header, meta=meta, use_dask=use_dask,
                   snidresult=snidresult, filename=filename,
                   load_snidres=load_snidres, **kwargs)

    @classmethod
    def from_name(cls, targetname, as_spectra=False, use_dask=False,
                      load_snidres=True, **kwargs):
        """ """        
        fullpath = SPEC_DATAFILE[SPEC_DATAFILE["name"]==targetname]["fullpath"].values
        if len(fullpath)==0:
            warnings.warn(f"No spectra target names {targetname}")
            return None
        
        if len(fullpath) == 1:
            return cls.from_filename(fullpath[0], use_dask=use_dask, load_snidres=load_snidres,
                                         **kwargs)
        elif as_spectra:
            return Spectra.from_filenames(fullpath, use_dask=use_dask, load_snidres=load_snidres,
                                         **kwargs)
        else:
            return [cls.from_filename(file_, use_dask=use_dask, load_snidres=load_snidres,
                                         **kwargs)
                        for file_ in fullpath]
        
    # ================ #
    #    Method        #
    # ================ #
    def fetch_snidresult(self, warn_if_notexist=True):
        """ """
        if self.filename is None:
            raise AttributeError("Unknown filename for the given spectrum. Cannot fetch the corresponding snidresult")

        snidresult_file = self.filename.replace(".ascii","_snid.h5")
        if not os.path.isfile(snidresult_file):
            if warn_if_notexist:
                warnings.warn(f"snidres file does not exists {snidresult_file}")
            return None
        
        from pysnid.snid import SNIDReader
        return SNIDReader.from_filename(snidresult_file)
    
    def load_snidresult(self, phase=None, redshift=None,
                            force_fit=False, allow_fit=True, **kwargs):
        """ get and set """
        if force_fit:
            allow_fit = True
            
        snidresult = self.fetch_snidresult(warn_if_notexist=False)
        if (snidresult is None or force_fit) and allow_fit:
            snidresult = self.get_snidfit(phase=phase, redshift=redshift,
                                       **kwargs)
        self.set_snidresult( snidresult )
        
    # --------- #
    #  SETTER   #
    # --------- #
    def set_data(self, data):
        """ """
        self._data = data
        
    def set_header(self, header):
        """ """
        self._header = header
    
    def set_meta(self, meta):
        """ """
        self._meta = meta

    def set_snidresult(self, snidresult):
        """ """
        self._snidresult = snidresult
        
    # --------- #
    #  GETTER   #
    # --------- #
    def get_obsdate(self):
        """ """
        from astropy.time import Time        
        if "date" not in self.meta.index or self.meta["date"] is None:
            warnings.warn("Unknown date for the given spectrum")
            return None
        
        return Time(self.meta["date"])
    
    def get_phase(self, t0, z=None):
        """ """
        obsdate = self.get_obsdate()
        if obsdate is None:
            return None
        phase = obsdate.mjd-t0
        if z is not None:
            phase /=(1+z)
        return phase
    
    def get_snidfit(self, phase=None, redshift=None,
                        delta_phase=5, delta_redshift=None,
                        lbda_range=[4000, 8000], **kwargs):
        """ """
        import pysnid
        if self.filename is None:
            raise NotImplementedError("No filename stored (self.filename is None). No work around implemented")
            
        return pysnid.run_snid(self.filename, redshift=redshift, phase=phase,
                                   delta_phase=delta_phase, delta_redshift=delta_redshift,
                                   lbda_range=lbda_range, **kwargs)
    
    
    # --------- #
    # PLOTTER   #
    # --------- #        
    def show(self, ax=None, savefile=None, color=None, ecolor=None, ealpha=0.2, 
             show_error=True, zeroline=True, zcolor="0.7", zls="--", 
             zprop={}, fillprop={}, normed=False, offset=None,
             label=None, legend=None, **kwargs):
        """ 
        label: [string or None]
            label for the spectra. 
            use: label='_meta_' to see meta content.
            
        """

        if ax is None:
            import matplotlib.pyplot as mpl
            fig = mpl.figure(figsize=[6,4])
            ax = fig.add_axes([0.12,0.15,0.78,0.75])
        else:
            fig = ax.figure
        
        self._compute()
        prop = dict(zorder=3)
        coef = 1 if not normed else self.flux.mean()
        if offset is None:
            offset = 0

        if label == "_meta_":
            label = f"{self.meta['name']} | by {self.meta.instrument} on {self.meta.date.split('T')[0]}"
        elif label == "_meta_noname_":
            label = f"{self.meta.instrument} on {self.meta.date.split('T')[0]}"
            
        if label is not None and legend is None:
            legend=True
            
        _ = ax.plot(self.lbda, self.flux/coef + offset, label=label, color=color, **{**prop, **kwargs})
        if self.has_error() and show_error:
            if ecolor is None:
                ecolor = color
            ax.fill_between(self.lbda, (self.flux-self.error)/coef+ offset, (self.flux+self.error)/coef+ offset, 
                           facecolor=ecolor, alpha=ealpha, **{**prop,**fillprop})
            
        if zeroline:
            ax.axhline(0, color=zcolor,ls=zls, **{**dict(lw=1, zorder=1),**zprop} )
            
        ax.set_ylabel("Flux []"+ " -normed-" if normed else "")
        ax.set_xlabel(r"Wavelength [$\AA$]")
        if legend:
            ax.legend(loc="best", frameon=False, fontsize="small")
            
        return fig

    def show_snidresult(self, axes=None, savefile=None, label=None, **kwargs):
        """ shortcut to self.snidresult.show() """
        if self.snidresult is None:
            warnings.warn("snidres is not defined (None)")
            return self.show(ax=axes[0])
        
        return self.snidresult.show(axes=axes, savefile=savefile, label=label, **kwargs)
    
    # ------------ #
    #   Internal   #
    # ------------ #
    def _compute(self):
        """ """
        if not self._use_dask:
            return
        from dask.delayed import Delayed
        from dask import delayed
        if not type(self._data) == Delayed:
            return 
        self._data, self._header = delayed(list)([self._data, self._header]).compute()
        
    # ================ #
    #   Properties     #
    # ================ #
    # Baseline    
    @property
    def data(self):
        """ """
        return self._data
    
    @property
    def header(self):
        """ """
        return self._header
    
    @property
    def meta(self):
        """ """
        return self._meta

    @property
    def snidresult(self):
        """ """
        return self._snidresult
    
    # Derived
    @property
    def lbda(self):
        """ """
        return np.asarray(self.data["lbda"].values, dtype="float")
    
    @property
    def flux(self):
        """ """
        return np.asarray(self.data["flux"].values, dtype="float")
    
    def has_error(self):
        """ """
        return self.has_variance()
    
    def has_variance(self):
        """ """
        return "variance" in self.data
    
    @property
    def variance(self):
        """ """
        if not self.has_error():
            return None
        
        return np.asarray(self.data["variance"].values, dtype="float")
        
    @property
    def error(self):
        """ """
        if not self.has_error():
            return None
        
        return np.sqrt(self.variance)

    @property
    def filename(self):
        """ """
        if not hasattr(self,"_filename"):
            return None
        return self._filename
    
# =============== #
#                 #
#                 #
#                 #
# =============== #
class Spectra( object ):
    """ Spectrum Collection """
    def __init__(self, spectra, use_dask=False):
        """ 
        spectra: [list]
            list of spectrum
        """
        self.set_spectra(spectra)
        self._use_dask = use_dask
        
    @classmethod
    def from_filenames(cls, filenames, use_dask=False):
        """ """
        spectra = [Spectrum.from_filename(file_, use_dask=use_dask) for file_ in filenames]
        return cls(spectra, use_dask=use_dask)
    
    @classmethod
    def from_directory(cls, directory, contains=None, startswith=None, extension=".ascii", use_dask=False):
        """ """
        from glob import glob
        glob_format = "*" if not startswith else f"{startswith}*"
        if contains is not None:
            glob_format += f"{contains}*"
        if extension is not None:
            glob_format +=f"{extension}"
        spectrafiles = glob(os.path.join(directory, glob_format))
        
        return cls.from_filenames(spectrafiles, use_dask=use_dask)

    # ================ #
    #    Method        #
    # ================ #
    def set_spectra(self, spectra):
        """ """
        self._spectra = spectra
        
    def get_meta(self, rebuild=False):
        """ """
        if rebuild:
            metas = self.call_down("meta", False)
            return pandas.DataFrame(metas).astype({"date":"<M8[ns]"})

        return self.meta

    def get_target_spectra(self, name):
        """ """
        return self.__class__( list(np.asarray(self.spectra)[np.asarray(self.meta.query(f"name == '{name}'").index)]),
                                  use_dask=self._use_dask )
    
    def get_spectra(self, sortby="date"):
        """ """
        if sortby is not None:
            index = self.meta.sort_values("date").index
            return np.asarray(self.spectra)[index]
        
        return self.spectra
        
    def show(self, ax=None, sortby="date", topfirst=True, legendprop={}, **kwargs):
        """ """
        if ax is None:
            import matplotlib.pyplot as mpl
            fig = mpl.figure(figsize=[7,4])
            ax = fig.add_axes([0.12,0.15,0.78,0.75])
        else:
            fig = ax.figure

        spectra = self.get_spectra(sortby)
        nspec = len(spectra)
        
        for i, spec in enumerate(spectra):
            _ = spec.show(ax=ax, offset=(nspec-i) if topfirst else i, 
                          normed=True, color=f"C{i}", 
                              legend=False, **{**dict(label="_meta_"),**kwargs})

        ax.legend(**{**dict(loc="best", frameon=False, fontsize="small"), **legendprop})
        return fig

    def show_target(self, name, ax=None, **kwargs):
        """ """
        spectarget = self.get_target_spectra(name)
        return spectarget.show(ax=ax, **kwargs)
    
    # -------- #
    # INTERNAL #
    # -------- #
    def map_down(self, what, margs, *args, **kwargs):
        """ """
        return [getattr(spec, what)(marg, *args, **kwargs)
                for spec, marg in zip(self.spectra, margs)]
    
    def call_down(self, what, isfunc, *args, **kwargs):
        """ """
        if isfunc:
            return [getattr(spec,what)(*args, **kwargs) for spec in self.spectra]
        return [getattr(spec,what) for spec in self.spectra]    
    
    # ================ #
    #   Properties     #
    # ================ #
    @property
    def meta(self):
        """ """
        if not hasattr(self,"_meta"):
            self._meta = self.get_meta(rebuild=True)
        return self._meta
    
    # Baseline    
    @property
    def spectra(self):
        """ """    
        return self._spectra

    @property
    def filenames(self):
        """ """
        return self.call_down("filename", isfunc=False)
