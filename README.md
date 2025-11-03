# *FOR INTERNAL USE OF LEGOLAS ONLY*

WARNING: Only install this repo if you are trying to install LEGOLAS. Unless you are a
developer you are in the wrong place. You should *not* directly use or depend on
anything in this repo. Install LEGOLAS instead, following the instructions in its
documentation.

Assuming all dependencies are satisfied:

```bash
pip install --no-deps --no-build-isolation --config-settings=--global-option=ext-all-sms -v -e .
```

For a standalone install, first create an env with dependencies:

```bash
conda env create --file ./environment.yaml
conda activate internal-legolas-aev
```
