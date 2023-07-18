# TODO:
# - whenever a dataset constraints less parameters than the other datasets, a vertical/horizontal bar should show in the 2D area, along with the legend. I haven't been able of doing this is a simple way, and I think this is mostly due to GetDist, which doesn't account for a missmatch in the parameters between MCMC's.

# NOTES:
# - if two or more parameters constrained by one dataset are constrained by another as a single parameter, e.g. Ωb and Ωc constrained by the CMB but only Ωm shows up in SnIa, then manual intervention is required to sum the previous two parameters (Ωb and Ωc) into a single one (Ωm). This should require a trivial modification of this code.

# imports
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os

# specify the output folders with the output of Stan
# WARNING: if the number of parameters differs between MCMC's, the one with more parameters must be the first
folders = ["output/fQ-Lfree/BAO", "output/fQ-Lfree/CC", "output/fQ-Lfree/SS/ET", "output/fQ-Lfree/SS/LISA", "output/fQ-Lfree/SnIa"]
legends = ["BAO", "CC", "ET", "LISA", "SnIa"]

# specify the parameters that were constrained in each model
paramsperfolder = [["h", "Om"], ["h", "Om"], ["h", "Om"], ["h", "Om"], ["Om"]]
labelsperfolder = [["h", "\\Omega_m"], ["h", "\\Omega_m"], ["h", "\\Omega_m"], ["h", "\\Omega_m"], ["h"]]

# get 'MCSamples' object for each run
mcsamples = []
for i in range(0, len(folders)):
    folder  = folders[i]
    params  = paramsperfolder[i]
    labels  = labelsperfolder[i]
    ndim    = len(params)
    legend  = legends[i]

    # get chains for each run
    files = sorted(os.popen(f"find {folder} -type f -name '*_?.csv'").read().split())
    chains = len(files)

    # get the samples from each chain
    samples = len(pandas.read_csv(files[0], comment="#")[params[0]])
    flatsamples = np.empty([samples*chains, ndim])
    chainN = 0
    for file in files:
        data = pandas.read_csv(file, comment="#")

        for i in range(0, len(params)):
            flatsamples[chainN::chains, i] = data[params[i]]

        chainN += 1

    mcsamples.append(MCSamples(samples=flatsamples, names=params, labels=labels, label=legend))

# make the corner plot
g = plots.get_subplot_plotter()
g.triangle_plot(mcsamples, filled=True)
plt.show()
