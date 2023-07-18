# imports
import os
import sys
import pandas
import arviz as az
from numpy.random import normal
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from getdist import plots, MCSamples

# provide run-specific parameters
theta   = ("h", "Ob", "Oc")                   # parameters
labels  = ("h", "\\Omega_b", "\\Omega_c")     # label for each parameter
inits   = {"h": 0.7, "Ob": 0.25, "Oc": 0.05}  # initial conditions for each parameter
model   = "model/fQ-Lfree/CMB.stan"           # input Stan model file
output  = "output/fQ-Lfree/CMB"               # output folder (None if you don't want to save the run)
data    = ["data/CMB.csv"]                    # data file(s)
warmup  = 1000                                # nº of warmup steps
samples = 1000                                # nº of sampling steps
chains  = 4                                   # nº of chains to run in total, and in parallel (if possible)

# to avoid overwriting, exit if directory already exists
if os.path.exists(output):
    print("output directory already exists, refusing to overwrite pre-existing data")
    sys.exit(1)

# ensure output directory exists
if output:
    os.system(f"mkdir -p {output}")

# create a model instance
model = CmdStanModel(
    #compile="force",                    # force model recompilation
    stan_file=model,                     # Stan model file location
    cpp_options={
        "STAN_NO_RANGE_CHECKS": "TRUE",  # don't check for elements out of bounds
        "STAN_THREADS":         "TRUE",  # allow running multiple chains in parallel
        "STAN_CPP_OPTIMS":      "TRUE"   # optimizations recommended by the Stan development team
    }
)

# fetch data from the data file(s)
# WARNING: the name of the columns in the files that are provided must not overlap
# WARNING: the order of the data files must match the order of N1, N2,... in the Stan model file
i = 1
dic = {}
for file in data:
    header = pandas.read_csv(file, comment="#", nrows=0).columns.tolist()
    columns = pandas.read_csv(file, comment="#")
    dic[f"N{i}"] = len(columns[header[0]])
    for var in header:
        if len(columns[var]) == 1:
            dic[var] = columns[var][0]
        else:
            dic[var] = columns[var]
    i += 1

# run the MCMC
fit = model.sample(
    #show_console=True,     # print output from cmfstan (useful for debugging)
    output_dir=output,      # save output to specified folder
    data=dic,               # data file or a dictionary with entries matching the data variables
    iter_warmup=warmup,     # number of warmup steps
    iter_sampling=samples,  # number of sampling steps
    inits=inits,            # initial conditions
    chains=chains,          # number of chains to run in total
    parallel_chains=chains  # number of chains to run in parallel (if possible)
)

# print and save MCMC summary
print()
print("============================== fit summary ==============================")
summary = fit.summary()
print(summary)
if output:
    with open(f"{output}/summary.csv", "w") as file:
        file.write(summary.to_string())

# print and save MCMC diagnostics
print()
print("============================ fit diagnostics ============================")
diagnose = fit.diagnose()
if output:
    with open(f"{output}/diagnostics.txt", "w") as file:
        file.write(diagnose)
print(diagnose)

# show and save the traceplot
posterior = az.from_cmdstanpy(posterior=fit)
az.plot_trace(posterior, var_names=theta, compact=False, combined=False)
plt.tight_layout()
if output:
    plt.savefig(f"{output}/traceplot.png")
plt.show()

# show and save corner plot
samples = samples = [fit.stan_variable(i) for i in theta]
mcsamples = MCSamples(samples=samples, names=theta, labels=labels)
g = plots.get_subplot_plotter()
g.triangle_plot(mcsamples, filled=True)
if output:
    plt.savefig(f"{output}/corner.png")
plt.show()
