from scipy import stats


noise_model = {
    # SALT paramerers
    "x1": { "func": stats.beta.rvs,
            "kwargs": {"a":1.78, "b":793.7, "loc":0.03, "scale":66.4}
          }, 
    "c": { "func": stats.alpha.rvs,
            "kwargs": {"a":3.27e+00, "loc":1.71e-02, "scale":5.03e-02}
          },

    # derived

    "magobs": { "func": stats.beta.rvs,
            "kwargs": { "a":3., "b":600., "loc":0.03 , "scale":2.}
            },
    # Environments
    "mass": { "func": stats.alpha.rvs,
            "kwargs": { "a":3.58e+00, "loc":1.01e-01, "scale":1.55e-03}
            },
    "localcolor": { "func": stats.alpha.rvs,
                    "kwargs":{"a":3e-7, "loc": 0.01, "scale": 0.017}
                }
    }
    
                
