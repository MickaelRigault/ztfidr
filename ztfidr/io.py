import os
import warnings
import pandas
import numpy as np
IDR_PATH = os.getenv("ZTFIDRPATH", "./dr2")

__all__ = ["get_targets_data",
           "get_host_data",
           "get_localhost_data",           
           "get_autotyping"]
# ================== #
#                    #
#   TOP LEVEL        #
#                    #
# ================== #
def get_targets_data():
    """ """
    redshifts = get_redshif_data()[["redshift","redshift_err", "redshift_source"]]
    salt2params = get_salt2params()
    coords = get_coords_data()
    # merging
    data_ = pandas.merge(redshifts,salt2params, left_index=True, right_index=True,
                     suffixes=("","_salt"), how="outer")
    data_ = pandas.merge(data_, coords, left_index=True, right_index=True,
                         how="outer")
    # force limit to target to use
    # target_list = get_targetlist()
    # data_ = data_.loc[target_list]
    return data_

def get_localhost_data(local_nkpc=2, which="mag"):
    """ """
    hostlocal = get_host_local(nkpc=2, which="mag")
    hostcoords = get_host_coords()
    dlr = get_host_mags().xs("global", axis=1)["host_dlr"]
    hostcoords = hostcoords.merge(dlr, left_index=True, right_index=True)
    return hostcoords.merge(hostlocal, left_index=True, right_index=True,
                                how="left")

def get_host_data(local_nkpc=2, which="mag"):
    """ """
    hostmag = get_host_mags().xs("global", axis=1)
    hostcoords = get_host_coords()
    return hostcoords.merge(hostmag, left_index=True, right_index=True,
                                how="left")
                                

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

# Master List
def get_masterlist(load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ztfdr2_masterlist.csv")
    if not load:
        return filepath
    
    return pandas.read_csv(filepath, **kwargs)
    
# Redshifts
def get_redshif_data(load=True, index_col=0, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ztfdr2_redshifts.csv")
    if not load:
        return filepath
    
    data = pandas.read_csv(filepath, index_col=index_col, **kwargs)
    data.index.name = 'ztfname'
    return data

def get_snidauto_redshift(load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH,"tables",
                            "ancilliary_info","snidauto_redshift.csv")
    if not load:
        return filepath
    return pandas.read_csv(filepath, index_col=0, **kwargs)


# Coordinates
def get_coords_data(load=True, index_col=0, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ztfdr2_coordinates.csv")
    if not load:
        return filepath
    return pandas.read_csv(filepath, index_col=index_col, **kwargs)

# SALT
def get_salt2params(load=True, default=True, **kwargs):
    """ """
    filename = "ztfdr2_salt2_params.csv" if default else\
      "ztfdr2_salt2_params_phase-15to30_color-0.4to0.8.csv"
    filepath = os.path.join(IDR_PATH, "tables",
                                filename)

    if not load:
        return filepath
    
    return pandas.read_csv(filepath, **kwargs
                          ).rename({"z":"redshift"}, axis=1
                          ).set_index("ztfname")


# ================== #
#                    #
#   HOST             #
#                    #
# ================== #
def get_host_coords(load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ancilliary_info/host_photometry/ztfdr2_hostcoords.csv")
    if not load:
        return filepath
    
    return pandas.read_csv(filepath, **{**dict(sep=" "),**kwargs}).set_index("ztfname")

def get_host_local(nkpc=2, which="mag", load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            f"ancilliary_info/host_photometry/ztfdr2_local{nkpc}kpc_{which}.csv")
    if not load:
        return filepath
    if not os.path.isfile(filepath):
        raise IOError(f"no such file {filepath}")

    return pandas.read_csv(filepath, index_col=0, **kwargs)

def get_host_sedfit(nkpc=2, load=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            f"ancilliary_info/host_properties/ztfdr2_local{nkpc}kpc_sedfit.csv")
    if not load:
        return filepath
    if not os.path.isfile(filepath):
        raise IOError(f"no such file {filepath}")

    return pandas.read_csv(filepath, index_col=0, **kwargs)
    

def get_host_mags(load=True, index_col=0, raw=False, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "ancilliary_info/host_photometry/ztfdr2_hostmags.csv")
    if not load:
        return filepath

    host_alldata = pandas.read_csv(filepath, index_col=index_col, **kwargs)
    if raw:
        return host_alldata

    def _get_split_(which):
        requested = f"{which}_mag"
        if requested not in host_alldata:
            raise ValueError(f"Unknown entry: {requested} (which={which})")

        host_data = host_alldata[requested].astype(str).str.replace("nan", "np.nan").str.replace("inf", "np.nan").apply(eval)
        host_err = host_alldata[requested+"_err"].astype(str).str.replace("nan", "np.nan").str.replace("inf", "np.nan").apply(eval)
        flagna = host_data.isna() + host_err.isna()
        data = pandas.DataFrame(list(host_data[~flagna]), index=host_data[~flagna].index)
        error = pandas.DataFrame(list(host_err[~flagna]), index=host_err[~flagna].index)
        error.columns += "_err"
        return pandas.merge(data, error, left_index=True, right_index=True)

    kpc2 = _get_split_(which="local_2kpc")
    kpc4 = _get_split_(which="local_4kpc")
    host_cat = _get_split_(which="host_cat")
    hglobal = pandas.merge(host_cat,
                          host_alldata[["z","host_dlr"]].rename({"z":"redshift"}, axis=1), 
                          left_index=True, right_index=True,
                               how="outer")
    return pandas.concat([kpc2, kpc4, hglobal], axis=1,
                          keys=["2kpc", "4kpc", "global"])
    
# ================== #
#                    #
#   LIGHTCURVES      #
#                    #
# ================== #
def get_target_lc(target, test_exist=True):
    """ """
    fullpath = os.path.join(IDR_PATH, "lightcurves", f"{target}_LC.csv")
    if test_exist:
        if not os.path.isfile(fullpath):
            warnings.warn(f"No lc file for {target} ; {fullpath}")
            return None

    return fullpath


def get_phase_coverage(load=True, warn=True, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables", "phase_coverage.csv")
    if not load:
        return filepath

    if not os.path.isfile(filepath):
        if warn:
            warnings.warn(
                "No phase_coverage file. build one using ztfidr.Sample().build_phase_coverage(store=True)")
        return None

    return pandas.read_csv(filepath, index_col=0, **kwargs
                           ).reset_index().rename({"index": "name"}, axis=1
                                                  ).set_index(["name", "filter"])["phase"]


# ================== #
#                    #
#   Spectra          #
#                    #
# ================== #
def get_autotyping(load=True, index_col=0, **kwargs):
    """ """
    filepath = os.path.join(IDR_PATH, "tables",
                            "autotyping.csv")
    if not load:
        return filepath
    return pandas.read_csv(filepath, index_col=index_col, **kwargs)


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
        glob_format += f"{extension}"

    specfiles = glob(os.path.join(IDR_PATH, "spectra", glob_format))
    datafile = pandas.DataFrame(specfiles, columns=["fullpath"])
    datafile["basename"] = datafile["fullpath"].str.split(
        "/", expand=True).iloc[:, -1]

    return pandas.concat([datafile, parse_filename(datafile["basename"], snidres=snidres)], axis=1)


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
