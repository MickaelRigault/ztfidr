[![PyPI](https://img.shields.io/pypi/v/ztfidr.svg?style=flat-square)](https://pypi.python.org/pypi/ztfidr)

# ztfidr
package to read and parse the ZTF SN IDR dataset

# Install
 `pip install ztfidr` [![PyPI](https://img.shields.io/pypi/v/ztfidr.svg?style=flat-square)](https://pypi.python.org/pypi/ztfidr)
 or (for the latest implementations)
 ```bash
 git clone https://github.com/MickaelRigault/ztfidr.git
 cd ztfidr
 python setup.py install
 ```
**AND**
you need to have clone the [ZTF Interal datarelease](https://github.com/ZwickyTransientFacility/ztfcosmoidr) (password protected).

**you need to provide the fullpath as a global environment variable using: $ZTFIDRPATH**

# Usage

Assuming your `ztfcosmoidr/dr2` repository is stored at a location provided by `$ZTFIDRPATH`:
```python
import ztfidr
sample = ztfidr.get_sample() # UPDATE FOR YOUR CASE
```

The dataframe containing all the relevant information is accessible as `sample.data`:
```python
sample.data
```
and `sample.get_data()` you have many options like `x1_range`, `t0_range`, `redshift_range` or `goodcoverage=True`...

<p align="left">
  <img src="images/example_data.png" width="900" title="data">
</p>


To visualise an individual target lightcurve (e.g. `ZTF19aampqcq`), do:
```python
lc = sample.get_target_lightcurve("ZTF19aampqcq")
lc.show()
```
<p align="left">
  <img src="images/example_show_target.png" width="900" title="show_target">
</p>



*More functionalities are implemented, check ztfidr.get_hosts_data() etc.*
