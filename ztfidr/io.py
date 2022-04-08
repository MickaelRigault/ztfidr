import os
import warnings
import pandas
import numpy as np
IDR_PATH = os.getenv("ZTFIDRPATH", "./dr2")


# ================== #
#                    #
#   DATA ACCESS      #
#                    #
# ================== #
    
# ================== #
#                    #
#   TARGET           #
#                    #
# ================== #
def get_target_lc(target, test_exist=True):
    """ """
    fullpath = os.path.join(IDR_PATH, "lightcurves",f"{target}_LC.csv")
    if test_exist:
        if not os.path.isfile(fullpath):
            warnings.warn(f"No lc file for {target} ; {fullpath}")
            return None
        
    return fullpath

def get_targets_data(merge_how="outer"):
    """ """
    redshift = get_redshif_data() 
    coords = get_coords_data()
    autotyping = get_autotyping()

    
    dd = pandas.merge(coords,redshift, left_index=True,
                          right_index=True, suffixes=("","_rt"),
                          how=merge_how)
    dd = pandas.merge(dd, autotyping, left_index=True,
                          right_index=True, suffixes=("","_rt"),
                          how=merge_how)
    
    return dd

def get_autotyping(load=True, index_col=0, **kwargs):
    """ """
    filepath =  os.path.join(IDR_PATH,"tables",
                             "autotyping.csv")
    if not load:
        return filepath
    return pandas.read_csv(filepath, index_col=index_col, **kwargs)

def get_redshif_data(load=True, index_col=0, **kwargs):
    """ """
    filepath =  os.path.join(IDR_PATH,"tables",
                             "ztfdr2_redshifts.csv")
    if not load:
        return filepath
    return pandas.read_csv(filepath, index_col=index_col, **kwargs)

def get_coords_data(load=True, index_col=0, **kwargs):
    """ """
    filepath =  os.path.join(IDR_PATH,"tables",
                             "ztfdr2_coordinates.csv")
    if not load:
        return filepath
    return pandas.read_csv(filepath, index_col=index_col, **kwargs)

def get_host_data(load=True, index_col=0, which=None, **kwargs):
    """ """
    filepath =  os.path.join(IDR_PATH,"tables",
                             "ancilliary_info/host_photometry/ztfdr2_hostmags.csv")
    if not load:
        return filepath
    
    host_alldata = pandas.read_csv(filepath, index_col=index_col, **kwargs)
    if which is None:
        return host_alldata

    requested = f"{which}_mag" 
    if requested not in host_alldata:
        raise ValueError(f"Unknown entry: {requested} (which={which})")

    host_data = host_alldata[requested]
    host_err  = host_alldata[requested+"_err"]
    data  = pandas.DataFrame( list(host_data.apply( eval ) ), index=host_data.index)
    error = pandas.DataFrame( list(host_err.apply( eval ) ), index=host_err.index)
    error.columns += "_err"
    return pandas.merge(data, error, left_index=True, right_index=True)    

# ================== #
#                    #
#   LIGHTCURVES      #
#                    #
# ================== #
def get_baseline_corrections(load=True):
    """ """
    filepath =  os.path.join(IDR_PATH,"lightcurves","baseline_corr",
                             f"DR2_Baseline_{band}band_corr.csv")
    if not load:
        return filepath
    return pandas.concat({band:pandas.read_csv(filepath, index_col=0)
                            for band in ["g","r","i"]})

def get_salt2params(load=True, default=True, **kwargs):
    """ """
    
    filename = "ztfdr2_salt2_params_phase-15to30_color-0.4to0.8.csv"

    filepath = os.path.join(IDR_PATH,"tables", filename)
    
    if not load:
        return filepath
    return pandas.read_csv(filepath, **kwargs).set_index("ztfname")

def get_phase_coverage(load=True, warn=True, **kwargs):
    """ """
    filepath =  os.path.join(IDR_PATH, "tables", "phase_coverage.csv")
    if not load:
        return filepath
    
    if not os.path.isfile(filepath):
        if warn:
            warnings.warn("No phase_coverage file. build one using ztfidr.Sample().build_phase_coverage(store=True)")
        return None
    
    return pandas.read_csv(filepath, index_col=0, **kwargs
                          ).reset_index().rename({"index":"name"}, axis=1).set_index(["name","filter"])["phase"]

    

# ================== #
#                    #
#   Spectra          #
#                    #
# ================== #
def get_spectra_datafile(contains=None, startswith=None,
                        snidres=False, extension=None, use_dask=False):
    """ """
    from glob import glob
    glob_format = "*" if not startswith else f"{startswith}*"
    if snidres and extension is None:
        extension = "_snid.h5"
    elif extension is None:
        extension = ".ascii"
        
    if contains is not None:
        glob_format += f"{contains}*"
    if extension is not None:
        glob_format +=f"{extension}"
        
        
    specfiles = glob(os.path.join(IDR_PATH, "spectra", glob_format))
    datafile = pandas.DataFrame(specfiles, columns=["fullpath"])
    datafile["basename"] = datafile["fullpath"].str.split("/",expand=True).iloc[:,-1]
    
    return pandas.concat([datafile, parse_filename(datafile["basename"], snidres=snidres) ], axis=1)

    
def parse_filename(file_s, snidres=False):
    """ file or list of files. 
    Returns
    -------
    Serie if single file, DataFrame otherwise
    """
    
    index = ["name", "date", "telescope", "origin"]
    fdata = []
    for file_ in np.atleast_1d(file_s):
        file_ = os.path.basename(file_).split(".ascii")[0]
        if snidres:
            #print(file_)
            name, date, *telescope, origin, snid_ = file_.split("_")
        else:
            try:
                name, date, *telescope, origin = file_.split("_")
            except:
                print(f"failed for {file_}")
                continue
            
        telescope = "_".join(telescope)
        fdata.append([name, date, telescope, origin])
    
    if len(fdata) == 1:
        return pandas.Series(fdata[0], index=index)
    return pandas.DataFrame(fdata, columns=index)
