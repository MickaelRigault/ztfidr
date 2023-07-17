import pandas
from ztfidr import io
import numpy as np


import os


_ORIGINAL_PATH = os.path.abspath( os.path.join(io.IDR_PATH,"../.."))

ZTF_HOST_REPO = os.path.join(_ORIGINAL_PATH, "ztfdr2_host_visualisation/cutouts/")
host_globalfile = os.path.join(io.IDR_PATH, "tables/.dataset_creation/host_prop/host_photometry/ztfdr2_global_imcat_flux.csv")

HOSTDATA  = pandas.read_csv(host_globalfile, index_col=0)


WAVEEFF={'ps1::g': 4866.46,
         'ps1::r': 6214.62,
         'ps1::i': 7544.57,
         'ps1::z': 8679.47,
         'ps1::y': 9633.25,
         'sdss::u': 3594.33,
         'sdss::g': 4717.6,
         'sdss::r': 6186.8,
         'sdss::i': 7506.21,
         'sdss::z': 8918.3,
         '2mass::j': 12410.29,
         '2mass::h': 16513.45,
         '2mass::ks': 21656.09,
         'galex::fuv':1516,
         'galex::nuv':2267,
          }
    
markers= {"ps1":"s",
          "sdss":"o",
          "2mass":"d",
          "galex":"H"}





def jansky_to_abmag(flux, flux_err):
    """
    $m_{\text{AB}}=-2.5\log _{10}f_{\nu }+8.90.$
    """
    
    mag = -2.5*np.log10(flux) + 8.90
    if flux_err is None:
        dmag = np.NaN
    else:
        dmag = +2.5/np.log(10) * np.abs(flux_err / flux)
            
    return mag, dmag
    



def get_target_hostpng(targetname, test_exists=True):
    """ """
    paths = {"cutout1":os.path.join(ZTF_HOST_REPO, f"host_match/{targetname}_zoom1_host_ls.png"),
             "cutout6":os.path.join(ZTF_HOST_REPO, f"host_match/{targetname}_zoom6_host_ls.png"),
             "photimg":os.path.join(ZTF_HOST_REPO, f"host_phot/images/{targetname}/global_PS1_r.jpg")
            }
    if test_exists:
        paths = {k:v if os.path.isfile(v) else None for k,v in paths.items()}
    return paths
    

def get_target_globalphotometry(targetname, add_mag=True):
    """ """    
    datadict= {bandname:[WAVEEFF.get(f"{bandname}", np.NaN), 
                          HOSTDATA.loc[targetname][f"{band}"],
                          HOSTDATA.loc[targetname][f"{band}_err"],
                         ] 
                for bandname, band in zip( ["ps1::g","ps1::r","ps1::i","ps1::z","ps1::y",
                                            "sdss::u","sdss::g","sdss::r","sdss::i","sdss::z",
                                            "2mass::j" ,"2mass::h","2mass::ks",
                                            "galex::fuv", "galex::nuv"],
                                           ["PS1g", "PS1r", "PS1i", "PS1z", "PS1y", 
                                            "SDSSu","SDSSg","SDSSr","SDSSi","SDSSz",
                                            "2MASSJ","2MASSH","2MASSKs",
                                           "GALEXFUV","GALEXNUV"]
                                          )
               }
    data = pandas.DataFrame.from_dict(datadict, orient="index", columns=["waveeff","flux","flux_err"])    
    if add_mag:
        mag, magerr = jansky_to_abmag(data["flux"],data["flux_err"])
        data["mag"] = mag
        data["mag_err"] = magerr
        
    return data


def show_host(targetname, inmag=True, fig=None):
    """ """
    hostimg = get_target_hostpng(targetname)
    data = get_target_globalphotometry(targetname).reset_index(names="filter").copy()
    
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    from matplotlib.image import imread
    if hostimg["cutout1"] is not None:
        img1 = imread(hostimg["cutout1"])
        shape = np.asarray(img1.shape[:2])
        width = shape/6.
    else:
        img1 = None
        shape = None
    if hostimg["cutout6"] is not None:
        img6 = imread(hostimg["cutout6"])
    else:
        img6 = None
    
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=[7,2])

    # Image Patches
    ax1 = fig.add_subplot(143)
    if img1 is not None:
        ax1.imshow(img1)
        rect = Rectangle(shape/2.-width/2, width=width[0], height=width[1],
                    facecolor="None", edgecolor="k", lw=0.5)
        ax1.add_patch(rect)
    else:
        ax1.text(0.5,0.5, "no image", transform=ax1.transAxes, va="center", ha="center")

    ax2 = fig.add_subplot(144)
    if img6 is not None:
        ax2.imshow(img6)
    else:
        ax2.text(0.5,0.5, "no image", transform=ax2.transAxes, va="center", ha="center")

    [ax_.set(xticks=[], yticks=[]) for ax_ in fig.axes]

    # Global mag
    ax =  fig.add_subplot(121)

    key = "mag" if inmag else "flux"

    
    data["markers"] = data["filter"].str.split("::", expand=True)[0].apply(markers.get).values

    # remove NaN
    data = data[~data[key].isna()]

    
    for marker, indexes in data.groupby("markers").groups.items():
        ax.scatter(data.loc[indexes]["waveeff"], data.loc[indexes][key], marker=marker, 
                  facecolors=to_rgba("C1", 0.8), edgecolors="0.5", zorder=6)

        ax.errorbar(data.loc[indexes]["waveeff"], data.loc[indexes][key], 
                    yerr=data.loc[indexes][f"{key}_err"],
                    marker="None", ls="None", ecolor="0.5", zorder=5)


    if not inmag:
        ax.set_yticks([0])
        ax.axhline(0, ls="--", color="k", lw=0.5)
        label = "flux [Jy]"
    else:
        ax.invert_yaxis()
        ylim = ax.get_ylim()
        ax.set_ylim( np.min([23, ylim[0]]) ,np.max([ylim[1],13]))
        label = "mag [AB]"

    ax.tick_params(labelsize="small")

    ax.set_xticks(np.linspace(1_000, 23_000, 5))
    ax.set_ylabel(label)
    ax.set_xlabel("wavelength")

    # fancy
    fig.tight_layout()
    return fig

