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

# Usage

Assuming your ZTFIDR repository is stored at “~/data/idr" then:
```python
import ztfidr
sample = ztfidr.load_idr(“~/data/idr/dr2") # UPDATE FOR YOUR CASE
```
The summary table is accessible as `sample.data`, the spectra as `sample.spectra` and the lightcurves as `sample.lightcurves`.
```python
sample.data
```
<p align="left">
  <img src="images/example_data.png" width="900" title="data">
</p>


To visualise an in individual target (e.g. `ZTF19aampqcq`), do:
```python
_ = sample.show_target("ZTF19aampqcq")
```
<p align="left">
  <img src="images/example_show_target.png" width="900" title="show_target">
</p>
The ticks on top of the lightcurve figure show the time where spectra have been taken, following the color coding of the spectral panel.


`sample.spectra` and `sample.lightcurves` have additional functionalities, for instance `sample.spectra.meta` contains the summary information of the spectral observations (date, telescope, etc.).


*More functionalities will be added as measure as the new information are added (host, lightcurve fit and so one)*
