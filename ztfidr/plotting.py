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
                                zref = 0.204, lw=2, rotation=9):
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
        data["mag"] = -2.5 * np.log10( data["x0"] ) + 19*1.58
        
    sc = ax.scatter(data["redshift"], 
                    data["mag"], c=data[param], 
                    cmap=cmap,
                    s=10, vmin=vmin, vmax=vmax)
    
    _ = utils.hist_colorbar(data[param], ax=cax, cmap=cmap, fcolorbar=0.1,
                           vmin=vmin, vmax=vmax, bins=bins)

    xx = np.linspace(0.00001,0.4,1000)
    ax.plot(xx, Planck18.distmod(xx), color="k", lw=lw)

    ax.set_ylabel("distance modulus + cst", fontsize="large")
    ax.set_xlabel("redshift", fontsize="large")
    cax.set_xlabel(f"{param}", fontsize="small", color="0.5")
    cax.tick_params(axis="x", labelsize="small", 
                   labelcolor="0.5", color="0.5")

    ax.text(zref, Planck18.distmod(zref).value, "Planck H(z)", va="center", ha="center", 
           fontsize="x-small", rotation=rotation, weight="bold",
            bbox={"facecolor":"w", "edgecolor":"None"})
    
    if clear_axes:
        clearwhich = ["bottom","right","top","left"] # "bottom"    
        [ax.spines[which].set_visible(False) for which in clearwhich]
        ax.tick_params(labelsize="medium")
#    ax.set_xticks([])

    return fig

# ================== #
#                    #
#   Lightcurves      #
#                    #
# ================== #
def show_npoints_distribution_perband(sample_or_phase_coverage, ax=None, phase_coverage=None, 
                                      clearaxes=True, add_pantheon=True):
    """ """
    if type(sample_or_phase_coverage) is pandas.DataFrame:
        phase_coverage = sample_or_phase_coverage
    else:
        phase_coverage = sample_or_phase_coverage.get_phase_coverage()


    from matplotlib.colors import to_rgba
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=[6, 3.5])
        axb = fig.add_axes([0.08, 0.15, 0.87, 0.35])
        axt = fig.add_axes([0.08, 0.6, 0.87, 0.35])

    else:
        fig = ax.figure
        axb, axt = fig.axew

    prop = dict(bins=np.logspace(0,3, 15), density=False, log=False)

    # -------------
    weights = np.ones(len(phase_coverage))*1
    axt.hist(phase_coverage["n_early_points_ztfg"]+phase_coverage["n_late_points_ztfg"], histtype="step", 
            fill=True, 
            edgecolor=to_rgba("tab:green", 1), lw=1., ls="--",
            facecolor=to_rgba("tab:green", 0.05), 
            label=f"ztf:g", weights=weights,
            zorder=2, **prop)

    axt.hist(phase_coverage["n_early_points_ztfr"]+phase_coverage["n_late_points_ztfr"], histtype="step", 
            fill=True, 
            edgecolor=to_rgba("tab:red", 1), lw=1., ls="-",
            facecolor=to_rgba("tab:red", 0.05), 
            label=f"ztf:r",weights=weights,
            zorder=3, **prop)

    axt.hist(phase_coverage["n_early_points_ztfi"]+phase_coverage["n_late_points_ztfi"], histtype="step", 
            fill=True, 
            edgecolor=to_rgba("tab:orange", 1), lw=1., ls=":",
            facecolor=to_rgba("tab:orange", 0.05), 
            label=f"ztf:i",weights=weights,
            zorder=5, **prop)


    # --------------
    weights = np.ones(len(phase_coverage))*1

    axb.hist(phase_coverage["n_points"], histtype="step", color="k", lw=1.5,
            label=f"any phase",
            weights=weights,
            zorder=5, **prop)

    axb.hist(phase_coverage["n_early_points"], histtype="step", 
            fill=True, 
            edgecolor=to_rgba("0.5", 1), lw=1., ls="-",
            facecolor=to_rgba("0.5", 0.), 
            label="pre-max",
            weights=weights,            
            zorder=3, **prop)

    axb.hist(phase_coverage["n_late_points"], histtype="step", 
            fill=True, 
            edgecolor=to_rgba("0.5", 1), lw=0., 
            facecolor=to_rgba("0.5", 0.3), 
            label="post-max",
            weights=weights,
            zorder=4, **prop)

    # --------------

    axb.set_xlabel("number of detected point", fontsize="large")
    axb.set_xscale("log")
    axt.set_xscale("log")

    if clearaxes:
        clearwhich = ["left","right","top"] # "bottom"
        for ax in [axb, axt]:
            [ax.spines[which].set_visible(False) for which in clearwhich]
            ax.tick_params(axis="y", labelsize="small", 
                       labelcolor="0.5", color="0.5")

    axt.legend(fontsize="medium", frameon=False)#, loc="upper left")
    axb.legend(fontsize="medium", frameon=False)

    axt.set_xlim(*axb.get_xlim())
    return fig


def show_firstdet_distributions(sample, ax=None, restrict_to=[-25,60]):
    """ """
    phases = sample.get_phases(phase_range=restrict_to)
    goodlc = sample.get_data(goodcoverage=True, 
                             x1_range=[-4,4], 
                             c_range=[-0.3,0.8],
                             t0_err_range=[0,1],
                             x1_err_range=[0,1], 
                             c_err_range=[0,0.1],
                            ).index.astype(str)

    from matplotlib.colors import to_rgba
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=[6, 2.5])
        ax = fig.add_axes([0.08, 0.1, 0.87, 0.8])
    else:
        fig = ax.figure

    first_det_per_band = phases.groupby(level=[0,1]).min()
    first_det = first_det_per_band.unstack().min(axis=1)

    prop = dict(range=[-21,30], bins=50, histtype="step", fill=True)

    # ---- Any band

    _ = ax.hist(first_det, facecolor="None", zorder=5,
                color="0.5", label="all targets", **prop)

    _ = ax.hist(first_det.loc[goodlc], zorder=4,
                facecolor="darkkhaki", lw=0,
                 label="‘Basic cuts’", **prop)
    # ---- Per filter

    #ax.legend(frameon=False)

    for band, color, label in zip(["ztfg","ztfr","ztfi"],
                          ["tab:green","tab:red","tab:orange"],
                          ["ztf:g","ztf:r","ztf:i"]):
        band_fdet = first_det_per_band.xs(band, level=1)
        band_fdet_good = band_fdet.loc[band_fdet.index.isin(goodlc)]


        _ = ax.hist(band_fdet_good, 
                    facecolor=to_rgba(color, 0.0), #zorder=2,
                    edgecolor=color,
                    weights = np.ones(len(band_fdet_good))*-1, 
                    label=label, **prop)

    ax.spines["bottom"].set_position(('data',0))
    ax.spines["bottom"].set_zorder(10)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks, labels=[f"{int(np.abs(y_))}" for y_ in yticks])
    
    
    clearwhich = ["left","right","top"] # "bottom"
    [ax.spines[which].set_visible(False) for which in clearwhich]
    ax.tick_params(axis="y", labelsize="small", 
               labelcolor="0.5", color="0.5")
    ax.tick_params(axis="x", pad=2, zorder=10)
    ax.set_xlabel("First detection phase [days]", loc="right")


    #ax.legend(["all", "good"], frameon=False)
    ax.legend(frameon=False, ncol=3)
    return fig


from pprint import pprint
import numpy as np


def show_residual(data, key="pull", ax=None, axh=None, axc=None,
                  binstat="mean", bins=np.arange(-50,50, 0.25),
                  main_color="C0", cmap_significance="Greys_r",
                  incl_lef=True, incl_runningpull=False,
                  pull_color=None,
                  blw=0.1, bs=8, bec="w",
                  **kwargs):
    """ """
    from matplotlib.colors import to_rgba
    from scipy import stats
    
    if ax is None:
        fig = plt.figure(figsize=[7,3])
        ax = fig.add_axes([0.1,0.2,0.7,0.7])
        axh = fig.add_axes([0.801,0.2,0.1,0.7])
        if cmap_significance is not None:
            axc =  fig.add_axes([0.801,0.9,0.1,0.02])
    else:
        fig = ax.figure

    prop = dict(s=0.2, lw=0, facecolors=main_color, edgecolor="None", zorder=2)
        
    data = data.copy() # not changing the input dataframe
    ax.scatter(data["phase"], data[key],**{**prop,**kwargs})
    if axh is not None:
        axh.hist(data[key], orientation="horizontal", range=[-6,6], bins="auto",
                 density=True, 
                 facecolor=to_rgba(main_color, 0.5),
                 edgecolor=to_rgba(main_color, 0.8),
                 histtype="step", fill=True, lw=1, zorder=1)
        axh.set_xticks([])
        axh.set_yticks([])
    
    lbins = 0.5*(bins[1:] + bins[:-1])
    data["phasebins"] = pandas.cut(data["phase"], bins, labels=lbins)
    bin_stats = data.groupby("phasebins")[key].agg([binstat, "std", "size", utils.nmad])
    
    
    prop_binstat = dict(s=bs, lw=blw, zorder=8, edgecolor=bec)
    if cmap_significance is not None:
        cmap = plt.get_cmap(cmap_significance, 5)
        significance = bin_stats[binstat] / (bin_stats["std"]/np.sqrt(bin_stats["size"]-1))
        if incl_lef:
            print("include look elsewhere effect")
            ntrials = len(lbins)
            proba = stats.norm.sf(np.abs(significance.values), loc=0, scale=1)*2 # x2 because two sides the same
            proba[proba<1e-15] = 1e-15
            proba_lef = np.asarray( 1- (1-proba)**ntrials, dtype="float64")
            significance = np.abs(-stats.norm.ppf(proba_lef/2)) # abs to avoid -0
            bin_stats["significance"] = significance
        else:
            bin_stats["significance"] = significance.abs()
            
        prop_binstat = {**prop_binstat, 
                        **dict(c=bin_stats["significance"], vmax=5, vmin=0, cmap=cmap)
                       }
        
    else:
        cmap = None
        prop_binstat = {**prop_binstat, 
                        **dict(facecolors="w")
                       }
        
    sc = ax.scatter(bin_stats.index, bin_stats[binstat], 
               **prop_binstat)

    if incl_runningpull:
        if pull_color is None:
            pull_color = to_rgba("w",0.3)
        ax.fill_between(bin_stats.index, bin_stats["nmad"], -bin_stats["nmad"],
                            color=pull_color, lw=1, zorder=5)
        
    if cmap is not None and axc is not None:
        fig.colorbar(sc, cax=axc, orientation="horizontal")

    if key in ["pull","pull_corr"]:
        ax.axhspan(-1,1, color=main_color, alpha=0.05, zorder=1)
        ax.axhspan(-2,2, color=main_color, alpha=0.05, zorder=1)
        ax.axhspan(-3,3, color=main_color, alpha=0.05, zorder=1)
        ax.set_ylim(-6,6)
        if axh is not None:
            from scipy import stats 
            # pull hist
            xx = np.linspace(-6,6, 100)
            axh.plot(stats.norm.pdf(xx, loc=0, scale=1), 
                     xx, lw=0.5, color="k", ls="--")
#            axh.plot(stats.norm.pdf(xx, loc=0, scale=1.2), 
#                     xx, lw=1, color="k", ls="-")
            
    ax.set_xlim(-50,50)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("phase [in days]", fontsize="large")
    ax.set_ylabel(f"pull [$\sigma$]", fontsize="large")
    if axh is not None:
        axh.set_ylim(*ax.get_ylim())
        clearwhich = ["bottom","right","top","left"] # "bottom"    
        [axh.spines[which].set_visible(False) for which in clearwhich]

    return fig

def show_lc_residuals(data, fig=None, axes=None,
                      incl_lef=True, cmap_significance="Greys", 
                      binstat="mean", key="pull",
                      bincadence=[1,1,2], # g, r, i
                      incl_runningpull=True, incl_filtername=True,
                      incl_linelabel=True, 
                      scoef_i=3, incl_hist=True,
                      phase_lines = [-20, -15, -10, -5, 0, 10, 20, 30, 40, 50],
                      propaxes={},
                        **kwargs):
    """ """
    from matplotlib.colors import to_rgba
    # data
    data_g = data[data["filter"]=="ztfg"]
    data_r = data[data["filter"]=="ztfr"]
    data_i = data[data["filter"]=="ztfi"]

    
    left = propaxes.get("left", 0.1)
    bottom, height = propaxes.get("bottom",0.1), propaxes.get("height", 0.27)
    vspan, hspan = propaxes.get("vspan", 0.015),  propaxes.get("hspan", 0.001)
    width, hwidth = propaxes.get("width", 0.78),  propaxes.get("hwidth", 0.1)
    cvspan = propaxes.get("cvspan", 0.)
    print(cvspan)
    # -- Figure layout -- #
    if axes is None:
        if fig is None:
            fig = plt.figure(figsize=[7,5])
            
        # axes
        axg = fig.add_axes([left, bottom+2*(height+vspan), width, height], zorder=3)
        #
        axr = fig.add_axes([left, bottom+1*(height+vspan), width, height], zorder=3)
        #
        axi = fig.add_axes([left, bottom+0*(height+vspan), width, height], zorder=3)

        if incl_hist:
            ahg = fig.add_axes([left+width+hspan, bottom+2*(height+vspan), hwidth, height], zorder=1)    
            ahr = fig.add_axes([left+width+hspan, bottom+1*(height+vspan), hwidth, height], zorder=1)
            ahi = fig.add_axes([left+width+hspan, bottom+0*(height+vspan), hwidth, height], zorder=1)
        else:
            ahg, ahr, ahi = None, None, None

        axc = fig.add_axes([left, bottom+3*(height+vspan)+cvspan, 0.1, 0.01], zorder=3)
        
    else:
        if len(axes) == 3:
            axg, axr, axi = axes
            ahg, ahr, ahi = None, None, None
            axc = None
        elif len(axes) == 4:
            axg, axr, axi, axc = axes
            ahg, ahr, ahi = None, None, None
        elif len(axes) == 6:
            axg, axr, axi, ahg, ahr, ahi = axes
            axc = None
        elif len(axes) == 7:
            axg, axr, axi, ahg, ahr, ahi, axc = axes
        else:
            raise ValueError(f"cannot parse the input axes size {len(axes)}, 3, 4, 6 or 7 expected.")
        
        fig = axg.figure
        
    # 


    # -- plotting -- #

    prop = {**dict(binstat=binstat, bs=20, blw=0.2, incl_lef=incl_lef, 
               cmap_significance=cmap_significance, key=key,
                       incl_runningpull=incl_runningpull), **kwargs}

    _ = show_residual(data_g, ax=axg, axh=ahg, main_color="tab:green", 
                      bins=np.arange(-50,50, bincadence[0]), 
                      axc = axc, pull_color=to_rgba("w",0.4),
                      **prop
                     )
    _ = show_residual(data_r, ax=axr, axh=ahr,main_color="tab:red", 
                      bins=np.arange(-50,50, bincadence[1]), pull_color=to_rgba("w",0.4),
                      **prop
                     )

    _ = show_residual(data_i, ax=axi, axh=ahi, main_color="tab:orange",  
                      bins=np.arange(-50,50, bincadence[2]),
                      s=prop.pop("s",1)*scoef_i, pull_color=to_rgba("w",0.55),
                      bs=prop.pop("bs",8)*1.2,
                      **prop)

    # -- texts -- #
    if incl_filtername:
        proptext = dict(rotation=-90, va="center", ha="left", zorder=1)
        axg.text(1.005,.5, "ztf:g", color="tab:green", transform=axg.transAxes, **proptext) 
        axr.text(1.005,.5, "ztf:r", color="tab:red", transform=axr.transAxes, **proptext)
        axi.text(1.005,.5, "ztf:i", color="tab:orange", transform=axi.transAxes, **proptext)
    
    
    [ax_.set_yticks([-4,-2,0,2,4]) for ax_ in [axg, axr, axi]]
    # -- phase lines -- #
    if phase_lines is not None:
        [ax_.axvline(l, lw=0.5, color="0.8", zorder=1) for ax_ in [axg, axr, axi] for l in phase_lines]

        if incl_linelabel:
            _ = [axg.text(l, 5., l, va="bottom", ha="center", color="0.7", fontsize="small") for l in phase_lines]
    
    # -- layout -- #
    _ = [ax_.set_ylim(-5,5) for ax_ in fig.axes if ax_ is not axc]
    _ = [ax.tick_params(labelsize="medium") for ax in fig.axes]
    _ = [ax_.set_xticklabels([]) for ax_ in [axr, axg]]
    
    if axc is not None:
        axc.set_xticks([])
        prop_clabel = dict(fontsize="small", transform=axc.transAxes,  color="0.5")
        axc.text(-0.01, 0.5, "0 ", va="center", ha="right", **prop_clabel)
        axc.text(1.01,0.5, " 5", va="center", ha="left", **prop_clabel)
        axc.text(0.5,1.01, "significance", va="bottom", ha="center", **prop_clabel)
        
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


# ================== #
#                    #
#   Typings          #
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

def show_typingorigin(typings):
    """ """    
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[7,3])
    ax = fig.add_axes([0.15,0.2, 0.7,0.75])


    color=["tab:blue","beige", "0.8"]

    joined = typings.groupby(level=0).sum().sort_values(ascending=False)
    labels = joined.index.str.replace("pec-","pec\n")

    for i, category in enumerate(joined.index):
        d_ = typings.xs(category)[["auto","master","arbiter"]].cumsum()[::-1]
        label = d_.index
        _ = ax.bar(labels[i], d_, color=color,
                  label=None if i>0 else label)

    cbar = ax.bar(labels, joined.values, facecolor="None", edgecolor="k", lw=1)
    ax.bar_label(cbar, fmt='%d', color="k", padding=2, weight="bold", fontsize="small")

    clearwhich = ["right","top","left"] # "bottom"    
    [ax.spines[which].set_visible(False) for which in clearwhich]
    ax.tick_params(labelsize="large")

    _ = ax.set_yticks([])
    _ = ax.set_yscale("log")

    ax.legend(ncol=1, frameon=False, loc="upper right")
    return fig
