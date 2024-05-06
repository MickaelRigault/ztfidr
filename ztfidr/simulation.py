from scipy import stats


noise_model = {
    # SALT paramerers
    "x1": { "func": stats.alpha.rvs,
            "kwargs": {"a":3.46e+00, "loc":-1.35e-01, "scale":9.81e-01}
          },
    "c": { "func": stats.alpha.rvs,
            "kwargs": {"a":3.27e+00, "loc":1.71e-02, "scale":5.03e-02}
          },

    # derived
    "magobs": { "func": stats.alpha.rvs,
            "kwargs": { "a":2.45e+00, "loc":2.54e-02, "scale":3.18e-02}
            },
    # Environments
    "mass": { "func": stats.alpha.rvs,
            "kwargs": { "a":3.58e+00, "loc":1.01e-01, "scale":1.55e-03}
            },
    "localcolor": { "func": stats.alpha.rvs,
                    "kwargs":{"a":3e-7, "loc": 0.01, "scale": 0.017}
                }
    }
    
                
