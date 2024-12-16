# *FOR INTERNAL USE OF LEGOLAS ONLY*

Installation, assuming all dependencies are satisfied:

```bash
pip install --no-deps --no-build-isolation --config-settings=--global-option=ext -v -e .
```

For standalone testing first create an env with dependencies:

```bash
conda env create --file ./environment.yaml
conda activate internal-legolas-aev
```
