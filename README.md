# *FOR INTERNAL USE OF LEGOLAS ONLY*

Installation:

```bash
conda env create --file ./environment.yaml
conda activate internal-legolas-aev
pip install --no-deps --no-build-isolation --config-settings=--global-option=ext -v -e .
```
