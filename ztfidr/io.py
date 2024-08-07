import os
import warnings
import pandas
import numpy as np
IDR_PATH = os.getenv("ZTFIDRPATH", "./dr2")

__all__ = ["get_targets_data"]
    
# ================== #
#                    #
#   TOP LEVEL        #
#                    #
# ================== #
def get_targets_data(saltmodel="default"):
    """ 
    Returns
    -------
    pandas.DataFrame, string
        - dataframe containing targets parameters (incl lc fit parameters)
        - name of the lc fit model.
    """
    redshifts = get_redshif_data()[["redshift","redshift_err", "source"]]
    saltparams, saltmodel = get_saltparams(which=saltmodel)
    coords = get_coords_data()
    
    # merging
    data_ = pandas.merge(redshifts,saltparams, left_index=True, right_index=True,
                     suffixes=("","_salt"), how="outer")
    data_ = pandas.merge(data_, coords, left_index=True, right_index=True,
                         how="outer")
    
    # force limit to target to use
    target_list = get_targetlist()
    data_ = data_.loc[target_list["ztfname"]]

    # adding classification
    typing = get_target_typing()
    data_ = data_.join(typing[["sn_type", "sub_type"]], how="left")
    return data_, saltmodel

# ================== #
#                    #
#   BASICS           #
#                    #
# ================== #
def get_targetlist(load=True, **kwargs):
    """ official list of target to use for dr2 """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ztfdr2_targetlist.csv")
    if not load:
        return filepath
    
    return pandas.read_csv(filepath, **kwargs)

def get_target_typing(load=True, index_col=0, sep=" ",
                        from_datacreation=False,
                        clean=False, generic_typing=True,
                          **kwargs):
    """ """
    if not from_datacreation: # bases:
        filepath = os.path.join(IDR_PATH, "tables/ztfdr2_classifications.csv")
        sep = ","
    else:
        filepath = os.path.join(IDR_PATH, "tables/.dataset_creation/sample_def",
                            "ztfdr2_finaltypings.csv")
        sep = ","
    if not load:
        return filepath
    
    data = pandas.read_csv(filepath, index_col=index_col, sep=sep, **kwargs)
    if not from_datacreation:
        return data
        
    return data

# Master List
def get_masterlist(load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            ".dataset_creation/sample_def/ztfdr2_masterlist.csv")
    if not load:
        return filepath
    
    data = pandas.read_csv(filepath, **kwargs)
    return data["ztfname"].values.astype(str)

def get_nospectralist(load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            ".dataset_creation/sample_def/ztfdr2_target_nospectra.csv")
    if not load:
        return filepath
    
    data = pandas.read_csv(filepath, names=["ztfname"], **kwargs)
    return data["ztfname"].values.astype(str)


# Redshifts
def get_redshif_data(load=True, index_col=0, full=False, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ztfdr2_redshifts.csv")
    if not load:
        return filepath
    
    data = pandas.read_csv(filepath, index_col=index_col, **kwargs)
    data.index.name = 'ztfname'

    if not full: # get only redshift and source
        return data
    
    # or get them all:
    redshift_dir = os.path.join(IDR_PATH, "tables", ".dataset_creation/redshifts")
    redshifts = {}

    
    # Emission lines
    sedm_em = pandas.read_csv(os.path.join(redshift_dir,
                                            "ztfdr2_sedmhanii_redshift.csv"),
                                  index_col=0)
    sedm_em.columns = sedm_em.columns.str.replace("redshift","sedm")
    
    
    
    nosedm_em = pandas.read_csv(os.path.join(redshift_dir,
                                                    "ztfdr2_nonsedmhanii_redshift.csv"),
                                     index_col=0)
    nosedm_em.columns = nosedm_em.columns.str.replace("redshift","nonsedm")
    
    # SNID
    snidauto = pandas.read_csv(os.path.join(redshift_dir,
                                                    "ztfdr2_snid_redshifts.csv"),
                                     index_col=0)
    snidauto.columns = snidauto.columns.str.replace("redshift", "snid")
    
    # Host-Z
    hostz = pandas.read_csv(os.path.join(redshift_dir,
                                                    "ztfdr2_hostphot_redshifts.csv"),
                                     index_col=0)
    hostz = hostz[["z"]]
    hostz.columns = ["gal"]
    hostz["gal_err"] = 1e-5


    
    # override
#    override = pandas.read_csv(os.path.join(redshift_dir,
#                                                    "ztfdr2_override_redshifts.csv"),
#                                     index_col=0)
#    override.columns = ["override"]

    # merge them
    
    data = data.join(sedm_em, on="ztfname")
    data = data.join(nosedm_em, on="ztfname")
    data = data.join(snidauto, on="ztfname")
    data = data.join(hostz, on="ztfname")
 #   data = data.join(override, on="ztfname")
    data
    
    return data
    
# Coordinates
def get_coords_data(load=True, index_col=0, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ztfdr2_coordinates.csv")
    if not load:
        return filepath
    return pandas.read_csv(filepath, index_col=index_col, **kwargs)

# SALT
def get_saltparams(load=True, which="default", **kwargs):
    """ 

    Parameters
    ----------
    which: string
        name of the salt2 fit file to load. 
        If 'default': ztfdr2_salt2_params.csv is used
        
    Returns
    -------
    pandas.DataFrame, string
        - dataframe containing targets lc fit parameters
        - name of the lc fit model.
    """
    if which == "default":
        filename = "ztfdr2_salt2_params.csv"
    else:
        filename = os.path.join(".dataset_creation/lc_fits", which)

    filepath = os.path.join(IDR_PATH, "tables", filename)
    saltmodel = _parse_salt2filename(filepath)
    
    
    if not load:
        return filepath, saltmodel
    
    return pandas.read_csv(filepath, **kwargs
                          ).rename({"z":"redshift"}, axis=1
                          ).set_index("ztfname"), saltmodel


def _parse_salt2filename(filename):
    """ """
    basename = os.path.basename(filename)
    saltmodel, *version = basename.split("_")[1].split("params")[0].split("-")
    saltmodel = saltmodel.strip()
    version = None if len(version) ==0 else version[0].strip()
    if version is None:
        if saltmodel == "salt2": # so name is salt2 only:
            return f"{saltmodel} v=T21" # default version T21
        return saltmodel
    
    return f"{saltmodel} v={version}"

# ================== #
#                    #
#   HOST             #
#                    #
# ================== #
def get_globalhost_data(load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ztfdr2_globalhost_prop.csv")
    if not load:
        return filepath
    
    return pandas.read_csv(filepath, index_col=0)

def get_localhost_data(load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ztfdr2_local2kpc_prop.csv")
    if not load:
        return filepath
    
    return pandas.read_csv(filepath, index_col=0)


def get_globalhost_mag():
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            ".dataset_creation/host_prop/host_photometry/ztfdr2_valid_hostphot_mag.csv")
    return pandas.read_csv(filepath, index_col=0)
    

def get_localhost_mag():
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            ".dataset_creation/host_prop/host_photometry/ztfdr2_local2_hostphot_mag.csv")
    return pandas.read_csv(filepath, index_col=0)
    
# ================== #
#                    #
#   LIGHTCURVES      #
#                    #
# ================== #
def get_target_lightcurve(target, test_exist=True, load=True):
    """ """
    fullpath = os.path.join(IDR_PATH, "lightcurves", f"{target}_LC.csv")
    if test_exist:
        if not os.path.isfile(fullpath):
            warnings.warn(f"No lc file for {target} ; {fullpath}")
            return None
        
    if not load:
        return fullpath
    
    return pandas.read_csv(fullpath,  delim_whitespace=True, comment='#')


def get_target_lightcurve_baseline(target):
    """ """
    filepath = get_target_lightcurve(target, load=False)
    lcinfo = open(filepath, "r").read().splitlines()
    i_struct = [i for i, line in enumerate(lcinfo) if line.startswith("# ------")]
    cols, *data = [l.replace("#","").split() for l in lcinfo[i_struct[1]+2: i_struct[2]]]
    return pandas.DataFrame(data=data, columns=cols)


def get_phase_coverage(load=True, warn=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables", "phase_coverage.parquet")
    if not load:
        return filepath

    if not os.path.isfile(filepath):
        if warn:
            warnings.warn(
                "No phase_coverage file. build one using ztfidr.Sample().build_phase_coverage(store=True)")
        return None

    return pandas.read_parquet(filepath, **kwargs)#.set_index(["ztfname", "filter"])


# ================== #
#                    #
#   Spectra          #
#                    #
# ================== #
def get_autotyping(load=True, index_col=0, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            ".dataset_creation/sample_def/spec_class/autotyping.csv")
    if not load:
        return filepath
    return pandas.read_csv(filepath, index_col=index_col, **kwargs)


def get_lightcurve_datafile(contains=None):
    """ """
    from glob import glob
    glob_format = "*" if contains is None else f"*{contains}*"
    files = glob(os.path.join(IDR_PATH, "lightcurves", glob_format))
    datafile = pandas.DataFrame(files, columns=["fullpath"])
    datafile["basename"] = datafile["fullpath"].str.split(pat="/", expand=True).iloc[:, -1]
    datafile["ztfname"] = datafile["basename"].str.split(pat="_", expand=True).iloc[:, 0]
    return datafile
    
def get_spectra_datafile(contains=None, startswith=None,
                         snidres=False, extension=None, use_dask=False,
                         add_phase=True, data=None):
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
        glob_format += f"{extension}"

    specfiles = glob(os.path.join(IDR_PATH, "spectra", glob_format))
    datafile = pandas.DataFrame(specfiles, columns=["fullpath"])
    datafile["basename"] = datafile["fullpath"].str.split(pat="/", expand=True).iloc[:, -1]
    
    specfile = pandas.concat([datafile, parse_filename(datafile["basename"], snidres=snidres)], axis=1)
    
    if add_phase:
        from astropy.time import Time
        if data is None:
            from .sample import get_sample
            data = get_sample().data
            
        specfile["dateiso"] = Time(np.asarray(specfile["date"].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}"), dtype=str), format="iso").mjd
        specfile = specfile.join(data[["t0", "redshift"]], on="ztfname")
        specfile["phase_obs"] = (specfile.pop("dateiso")-specfile.pop("t0"))
        specfile["phase"] = specfile["phase_obs"]/(1+specfile.pop("redshift"))
        
    return specfile


def parse_filename(file_s, snidres=False):
    """ file or list of files.
    Returns
    -------
    Serie if single file, DataFrame otherwise
    """

    index = ["ztfname", "date", "telescope", "version"]
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
                print(f"failed parsing filename for {file_}")
                continue

        telescope = "_".join(telescope)
        fdata.append([name, date, telescope, origin])

    if len(fdata) == 1:
        return pandas.Series(fdata[0], index=index)
    return pandas.DataFrame(fdata, columns=index)
