import pandas
import numpy as np
import matplotlib.pyplot as plt

from . import utils

def show_hubble_standardisation(sample_or_data, fig=None,
                                param="c", vmin=-0.2, vmax=0.3, cmap="coolwarm", 
                                clear_axes=False, bins="auto"):
    """ """
    from astropy.cosmology import Planck18
    if not type(sample_or_data) == pandas.DataFrame:
        data = sample_or_data.get_data(x1_range=[-4,4], c_range=[-0.3,0.8], goodcoverage=True)
        data = data[data["classification"].isin(['snia-norm'])]
    else:
        data = sample_or_data.copy()
        
    if fig is None:
        fig = plt.figure(figsize=[6,4])
    
    ax = fig.add_axes([0.1,0.15,0.8,0.75])
    cax = fig.add_axes([0.4,0.3,0.4,0.25])

    if "mag" not in data:
        data["mag"] = -2.5 * np.log10(data["x0"])+19*1.58
        
    sc = ax.scatter(data["redshift"], 
                    data["mag"], c=data[param], 
                    cmap=cmap,
                    s=10, vmin=vmin, vmax=vmax)

    _ = utils.hist_colorbar(data[param], ax=cax, cmap=cmap, fcolorbar=0.1,
                           vmin=vmin, vmax=vmax, bins=bins)

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
    ia_norm = dist_typing[["snia-norm"]].sum()
    ia_pec = dist_typing[["snia-pec-91t","snia-pec-91bg","snia-pec"]].sum()
    ia = dist_typing[["snia"]].sum()
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
    ax.set_yticks([1,0,-1,-2], ["SN Ia\nnorm","SN Ia","SN Ia\npeculiar", "Unclear\nnon-ia"])

    return fig
