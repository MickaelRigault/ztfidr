import pandas
import numpy as np
import matplotlib.pyplot as plt

from . import utils

# ================== #
#                    #
#   SAMPLE           #
#                    #
# ================== #
def show_hubble_standardisation(sample_or_data, fig=None,
                                param="c", vmin=-0.2, vmax=0.3, cmap="coolwarm", 
                                clear_axes=False, bins="auto",
                                zref = 0.204):
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

    xx = np.linspace(0.00001,0.23,1000)
    ax.plot(xx, Planck18.distmod(xx), color="k", lw=2)

    ax.set_ylabel("distance modulus + cst", fontsize="large")
    ax.set_xlabel("redshift", fontsize="large")
    cax.set_xlabel(f"{param}", fontsize="small", color="0.5")
    cax.tick_params(axis="x", labelsize="small", 
                   labelcolor="0.5", color="0.5")

    ax.text(zref, Planck18.distmod(zref).value, "Planck H(z)", va="center", ha="center", 
           fontsize="x-small", rotation=9, weight="bold",
            bbox={"facecolor":"w", "edgecolor":"None"})
    
    if clear_axes:
        clearwhich = ["bottom","right","top","left"] # "bottom"    
        [ax.spines[which].set_visible(False) for which in clearwhich]
        ax.tick_params(labelsize="medium")
#    ax.set_xticks([])

    return fig

# ================== #
#                    #
#   Typing           #
#                    #
# ================== #
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

# ================== #
#                    #
#   Spectra          #
#                    #
# ================== #
def show_peakmag_barhist(sample_or_data, bts_data=None,
                            fig=None,
                            bins=np.linspace(12,22, 41),
                            facecolor_color="w", main_color="0.7",
                            filtername="g", xlim=[14,21]
                        ):
    """ 
    """
    if bts_data is None:
        bts_data = pandas.read_csv("https://sites.astro.caltech.edu/ztf/bts/explorer.php?f=s&subsample=all&classstring=&classexclude=&ztflink=lasair&lastdet=&startsavedate=&startpeakdate=&startra=&startdec=&startz=&startdur=&startrise=&startfade=&startpeakmag=&startabsmag=&starthostabs=&starthostcol=&startb=&startav=&endsavedate=&endpeakdate=&endra=&enddec=&endz=&enddur=&endrise=&endfade=&endpeakmag=19.0&endabsmag=&endhostabs=&endhostcol=&endb=&endav=&format=csv")
        
    bts_data = bts_data[bts_data["type"] != "-"] # secured typing in BTS
    
    if not type(sample_or_data) == pandas.DataFrame:
        data = sample_or_data.get_data()
    else:
        data = sample_or_data
    
    # Figure
    if fig is None:
        fig = plt.figure(figsize=[7,3])
    
    ax = fig.add_axes([0.10,0.15,0.8,0.75])
    
    data_inbts = data.loc[data.index.isin(bts_data["ZTFID"])]
    ampl, _ = np.histogram(data[f"peak_mag_ztf{filtername}"], bins)
    ampl_inbts, _ = np.histogram(data_inbts[f"peak_mag_ztf{filtername}"], bins)

    # DR2 hist
    ax.bar(bins[:-1], ampl, width=0.42/2, align="edge", 
           facecolor=facecolor_color, 
           edgecolor=main_color, 
           label=f"ZTF Cosmo DR2")

    # DR2 in BTS hist
    ax.bar(bins[:-1], ampl_inbts, width=0.42/2, align="edge", 
           facecolor=main_color, 
           edgecolor="None", 
           label=f"in BTS")

    ax.set_xticks(np.arange(bins[0], bins[-1], 1))

    ax.set_xlim(*xlim)
    ax.set_xlabel(f"SALT2 peak magnitude [ztf:{filtername}]", fontsize="medium")

    clearwhich = ["right","top", "left"] # "bottom"
    [ax.spines[which].set_visible(False) for which in clearwhich]
    ax.tick_params("y", labelsize="small")
    ax.legend(loc="upper left", frameon=False, ncol=1, fontsize="medium")

    return fig


def show_spectra_sources(specdata=None, bts_data=None, min_nspec=50, 
                            main_color = "0.7", facecolor_color = "w"):
    """ """
    # spectrum files    
    if specdata is None:
        from . import io
        specdata = io.get_spectra_datafile()
        
    # BTS data    
    if bts_data is None:
        bts_data = pandas.read_csv("https://sites.astro.caltech.edu/ztf/bts/explorer.php?f=s&subsample=all&classstring=&classexclude=&ztflink=lasair&lastdet=&startsavedate=&startpeakdate=&startra=&startdec=&startz=&startdur=&startrise=&startfade=&startpeakmag=&startabsmag=&starthostabs=&starthostcol=&startb=&startav=&endsavedate=&endpeakdate=&endra=&enddec=&endz=&enddur=&endrise=&endfade=&endpeakmag=19.0&endabsmag=&endhostabs=&endhostcol=&endb=&endav=&format=csv")
        
    bts_data = bts_data[bts_data["type"] != "-"] # secured typing in BTS
    # spectra in BTS
    specdata_inbts = specdata[ specdata["ztfname"].isin(bts_data["ZTFID"]) ]
    
    # local function
    def get_specsource_distribution(data_, min_number=min_nspec):
        df = data_.copy()
        ndata = df.groupby("telescope").size().sort_values(ascending=True)#.reset_index()
        as_others = ndata[ndata<=min_number].index
        df.loc[df["telescope"].isin(as_others),"telescope"] = "Other"
        df[df["telescope"].isin(as_others)]
        return df.groupby("telescope").size().sort_values(ascending=True).reset_index()
        #host_redshifts[host_redshifts["telescope_source"].isin(as_others)]["telescope_source"]
    
    
    
    ndatas = get_specsource_distribution(specdata).rename({0:"dr2"}, axis=1)
    ndatas_inbts = get_specsource_distribution(specdata_inbts).rename({0:"bts"}, axis=1)
    ndatas = ndatas.merge(ndatas_inbts, on="telescope", how="left").fillna(0)


    fig = plt.figure(figsize=[7,4])
    ax = fig.add_axes([0.12,0.15,0.8,0.75])

    hbars = ax.barh(ndatas.index, ndatas["dr2"], tick_label=ndatas["telescope"],
                   facecolor=facecolor_color, 
                    edgecolor=main_color, label="ZTF Cosmo DR2")

    hbars_bts = ax.barh(ndatas.index, ndatas["bts"], tick_label=ndatas["telescope"],
                   facecolor=main_color, edgecolor=main_color, label="in BTS")


    ax.tick_params("y", labelsize="small", color=main_color, length=0)
    ax.bar_label(hbars, fmt='%d', color=main_color, padding=2, fontsize="small")

    ax.legend(loc="lower right", frameon = False, ncol=1, fontsize="medium")
    clearwhich = ["bottom","right","top", "left"] # "bottom"
    [ax.spines[which].set_visible(False) for which in clearwhich]
    ax.set_xticks([])
    
    return fig


def show_phase_hist(specdata, sample_or_data=None, 
                    main_color = "darkkhaki",
                    second_color = None, xrange = [-22, 60], **kwargs):
    """ 
    """
    if second_color is None:
        second_color = main_color
        
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[7,2])
    ax = fig.add_axes([0.1,0.25,0.8,0.7])

    if sample_or_data is None:
        from . import get_sample
        sample_or_data = get_sample()

    
    if type(sample_or_data) == pandas.DataFrame:
        good_targets = sample_or_data.index
    else:
        good_targets = sample_or_data.get_data(goodcoverage=True, x1_range=[-4,4], c_range=[-0.3,0.8]).index


    prop = {**dict(bins=41, range=xrange, histtype="step", fill=True),
            **kwargs}
    
    _ = ax.hist(specdata["phase"], facecolor="None", edgecolor=main_color, 
                label="all targets", zorder=3,
                **prop)
    _ = ax.hist(specdata.query("ztfname in @good_targets")["phase"], 
                facecolor=second_color, edgecolor="None",
                label="after 'Basic cuts'",zorder=2,
                **prop)

    ax.set_xlim(*xrange)
    ax.set_xlabel("Phase [days]")
    ax.tick_params("y", labelsize="small", color="0.7", labelcolor="0.7")

    ax.legend(loc="upper right", frameon = False, ncol=1, fontsize="medium")
    clearwhich = ["right","top", "left"] # "bottom"
    [ax.spines[which].set_visible(False) for which in clearwhich]
    
    return fig
