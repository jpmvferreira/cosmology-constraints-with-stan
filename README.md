A set of cosmological models coded in the Stan programming language to be constrained using different observables. The motivation behind this repository is to provide a starting ground for people interested in constraining cosmological models using Stan.

The folder structure for this repository is as follows:
- `run.py`: A Python script that runs the MCMC, with the user provided parameters;
- `plots.py`: A Python script that plots the output of the MCMC's (also done automatically at the end of each run when using `run.py`);
- `data/`: The folder containing the datasets;
- `models/`: The folder that holds all the cosmological models to be constrained with a specific dataset, coded in Stan;
- `output/`: The output folder for the MCMC's.

All files should be documented in such a way that a new user will not feel completely lost when using them. For more information, refer to:
- The Stan documentation: [User Guide](https://mc-stan.org/docs/stan-users-guide/index.html), [Language Reference Manual](https://mc-stan.org/docs/reference-manual/index.html) and [Language Functions Reference](https://mc-stan.org/docs/functions-reference/index.html);
- [The CmdStanPy documentation](https://mc-stan.org/cmdstanpy/);
- [Getdist documentation](https://getdist.readthedocs.io/en/latest/).

To run these scripts you need Python version 3 installed in your system and the CmdStanPy, GetDist, Pandas and ArviZ Python packages, which are all available via pip or conda.
