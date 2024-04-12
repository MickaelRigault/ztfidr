import numpy as np
from scipy import stats


from astropy.cosmology import Planck18 as cosmo

def get_taylor_hostmass(magi, magg, magi_err, magg_err, redshift, 
                       sigma_int=0.1, cosmo=cosmo,
                       use_prior=False):
    """ computes the Taylor mass from given host magnitudes and redshifts.
    
    If use_prior=True, the g-i color gets a prior to avoid issues with low SNR 
    colors as in Rigault+2020.
    
    sigma_int is is intrinsic scatter of 0.1 dex from Taylor et al
    
    See eq. 8 of Taylor et al 2011 (2011MNRAS.418.1587T)
    ```
      log M∗/[M⊙] = 1.15 + 0.70(g − i) − 0.4Mi
    ```

    Returns
    -------
    mass, mass_err

    """
    color = magg-magi
    color_err = np.sqrt(magg_err**2 + magi_err**2)
    
    if use_prior:
        color, color_err = get_priored_color(color, color_err)

    magabs_i = magi - cosmo.distmod(np.asarray(redshift)).value
    
    mass = 1.15 + 0.70*color - 0.4*(magabs_i)
    mass_err = np.sqrt( (color_err*0.7)**2 + (0.4*magi_err**2) + sigma_int**2)
    
    return mass, mass_err
    
def get_priored_color(c, cerr, ndraw=2000):
    """ """
    c = np.atleast_1d(c)
    cerr = np.atleast_1d(cerr)
    
    xx, priorpdf = gi_prior_pdf(xx=f'-0.5:2.5:{int(ndraw*5)}j')
    likelihood = stats.norm.pdf(xx[:,None], loc=c, scale=cerr)
    posterior = likelihood*priorpdf[:,None] # at a constant
    
    draws = [np.random.choice(xx, size=ndraw, replace=True, p=p_/np.nansum(p_))
         for p_ in posterior.T]

    # assuming gaussian err
    new_c = np.mean(draws, axis=1)
    new_c_err = np.std(draws, axis=1) 
    
    return new_c, new_c_err

from scipy import stats
def gi_prior_pdf(mur=1.25, mub=0.85, sigmar=0.1, sigmab=0.3, b_coef=0.9,
                  xx="-0.5:2.5:1000j"):
    """ """
    if type(xx) == str:
        xx = eval(f"np.r_[{xx}]")
    pdf_b = stats.norm.pdf(xx, loc=mub, scale=sigmab)
    pdf_r = stats.norm.pdf(xx, loc=mur, scale=sigmar)
    return xx, pdf_b * b_coef + pdf_r * (1-b_coef)

