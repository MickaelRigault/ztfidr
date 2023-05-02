import numpy as np
import matplotlib.pyplot as plt

from . import utils

def show_hubble_standardisation(sample, fig=None,
                                param="c", vmin=-0.2, vmax=0.3, cmap="coolwarm", 
                                clear_axes=False):
    """ """
    from astropy.cosmology import Planck18
    data = sample.get_data(x1_range=[-4,4], c_range=[-0.3,0.8], goodcoverage=True)
    data = data[data["classification"].isin(['ia(-norm)','ia-norm'])]
    
    if fig is None:
        fig = plt.figure(figsize=[6,4])
    
    ax = fig.add_axes([0.1,0.15,0.8,0.75])
    cax = fig.add_axes([0.4,0.3,0.4,0.25])

    mag = -2.5 * np.log10(data["x0"])
    sc = ax.scatter(data["redshift"], 
                    mag+19*1.58, c=data[param], 
                    cmap=cmap,
                    s=10, vmin=vmin, vmax=vmax)


    _ = utils.hist_colorbar(data[param], ax=cax, cmap=cmap, fcolorbar=0.1,
                           vmin=vmin, vmax=vmax)

    xx = np.linspace(0.002,0.23,1000)
    ax.plot(xx, Planck18.distmod(xx), color="k", lw=2)

    ax.set_ylabel("distance modulus + cst", fontsize="large")
    ax.set_xlabel("redshift", fontsize="large")
    cax.set_xlabel(f"{param}", fontsize="small", color="0.5")
    cax.tick_params(axis="x", labelsize="small", 
                   labelcolor="0.5", color="0.5")
    zref = 0.204
    ax.text(zref, Planck18.distmod(zref).value, "Planck H(z)", va="center", ha="center", 
           fontsize="x-small", rotation=9, weight="bold",
            bbox={"facecolor":"w", "edgecolor":"None"})
    
    if clear_axes:
        clearwhich = ["bottom","right","top","left"] # "bottom"    
        [ax.spines[which].set_visible(False) for which in clearwhich]
        ax.tick_params(labelsize="medium")
#    ax.set_xticks([])

    return fig


def show_typingdistribution(sample, ax=None, fig=None):
    """ """
    dist_typing = sample.data.groupby("classification").size()
    ia_norm = dist_typing[["ia-norm","ia(-norm)"]].sum()
    ia_pec = dist_typing[["ia-91t","ia-91bg","ia-other"]].sum()
    ia = dist_typing[["sn ia"]].sum()
    rest = dist_typing.sum() - (ia_norm+ia_pec+ia)
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=[7,3])
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    ax.barh(1, ia_norm, facecolor="C0", edgecolor="None")
    ax.barh(0, ia, facecolor="0.7", edgecolor="None")
    ax.barh(-1, ia_pec, facecolor="C1", edgecolor="None")
    ax.barh(-2, rest, facecolor="None", edgecolor="k")
        
    ax.text(ia_norm+30,1, f'{ia_norm}', fontsize="large",
            color="C0", va="center", ha="left", weight="bold")
    ax.text(ia+30,0, f'{ia}', fontsize="large",
            color="0.7", va="center", ha="left", weight="bold")
    ax.text(ia_pec+30,-1, f'{ia_pec}', fontsize="large",
            color="C1", va="center", ha="left", weight="bold")

    ax.text(rest+30,-2, f'{rest}', fontsize="large",
            color="k", va="center", ha="left")

    clearwhich = ["bottom","right","top",] # "bottom"
    [ax.spines[which].set_visible(False) for which in clearwhich]
    ax.set_xticks([])
    ax.set_yticks([1,0,-1,-2], ["SN Ia\nnorm","SN Ia","SN Ia\npeculiar", "Unclear"])

    return fig
