"""
MCMC sampling using the emcee package
=====================================

Introduction
------------

The goal of Markov Chain Monte Carlo (MCMC) algorithms is to approximate
the posterior distribution of your model parameters by random sampling
in a probabilistic space. For most readers this sentence was probably
not very helpful so here we’ll start straight with and example but you
should read the more detailed mathematical approaches of the method
`here <https://www.pas.rochester.edu/~sybenzvi/courses/phy403/2015s/p403_17_mcmc.pdf>`__
and
`here <https://github.com/jakevdp/BayesianAstronomy/blob/master/03-Bayesian-Modeling-With-MCMC.ipynb>`__.

How does it work ?
~~~~~~~~~~~~~~~~~~

The idea is that we use a number of walkers that will sample the
posterior distribution (i.e. sample the Likelihood profile).

The goal is to produce a “chain”, i.e. a list of :math:`\theta` values,
where each :math:`\theta` is a vector of parameters for your model. If
you start far away from the truth value, the chain will take some time
to converge until it reaches a stationary state. Once it has reached
this stage, each successive elements of the chain are samples of the
target posterior distribution. This means that, once we have obtained
the chain of samples, we have everything we need. We can compute the
distribution of each parameter by simply approximating it with the
histogram of the samples projected into the parameter space. This will
provide the errors and correlations between parameters.

Now let’s try to put a picture on the ideas described above. With this
notebook, we have simulated and carried out a MCMC analysis for a source
with the following parameters: :math:`Index=2.0`,
:math:`Norm=5\times10^{-12}` cm\ :math:`^{-2}` s\ :math:`^{-1}`
TeV\ :math:`^{-1}`, :math:`Lambda =(1/Ecut) = 0.02` TeV\ :math:`^{-1}`
(50 TeV) for 20 hours.

The results that you can get from a MCMC analysis will look like this :

On the first two top panels, we show the pseudo-random walk of one
walker from an offset starting value to see it evolve to a better
solution. In the bottom right panel, we show the trace of each 16
walkers for 500 runs (the chain described previsouly). For the first 100
runs, the parameter evolve towards a solution (can be viewed as a
fitting step). Then they explore the local minimum for 400 runs which
will be used to estimate the parameters correlations and errors. The
choice of the Nburn value (when walkers have reached a stationary stage)
can be done by eye but you can also look at the autocorrelation time.

Why should I use it ?
~~~~~~~~~~~~~~~~~~~~~

When it comes to evaluate errors and investigate parameter correlation,
one typically estimate the Likelihood in a gridded search (2D Likelihood
profiles). Each point of the grid implies a new model fitting. If we use
10 steps for each parameters, we will need to carry out 100 fitting
procedures.

Now let’s say that I have a model with :math:`N` parameters, we need to
carry out that gridded analysis :math:`N*(N-1)` times. So for 5 free
parameters you need 20 gridded search, resulting in 2000 individual fit.
Clearly this strategy doesn’t scale well to high-dimensional models.

Just for fun: if each fit procedure takes 10s, we’re talking about 5h of
computing time to estimate the correlation plots.

There are many MCMC packages in the python ecosystem but here we will
focus on `emcee <https://emcee.readthedocs.io>`__, a lightweight Python
package. A description is provided here : `Foreman-Mackey, Hogg, Lang &
Goodman (2012) <https://arxiv.org/abs/1202.3665>`__.

"""

# %matplotlib inline
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.irf import load_irf_dict_from_file
from gammapy.maps import WcsGeom, MapAxis
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
    GaussianSpatialModel,
    SkyModel,
    Models,
    FoVBackgroundModel,
    GaussianPrior,
    UniformPrior,
)
from gammapy.datasets import MapDataset
from gammapy.makers import MapDatasetMaker
from gammapy.data import Observation

from gammapy.modeling import Fit

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


######################################################################
# Simulate an observation
# -----------------------
# 
# Here we will start by simulating an observation using the
# `simulate_dataset` method.
# 

irfs = load_irf_dict_from_file(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)

observation = Observation.create(
    pointing=SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic"),
    livetime=20 * u.h,
    irfs=irfs,
)

# Define map geometry
axis = MapAxis.from_edges(
    np.logspace(-1, 2, 15), unit="TeV", name="energy", interp="log"
)

geom = WcsGeom.create(
    skydir=(0, 0), binsz=0.05, width=(2, 2), frame="galactic", axes=[axis]
)

empty_dataset = MapDataset.create(geom=geom, name="dataset-mcmc")
maker = MapDatasetMaker(selection=["background", "edisp", "psf", "exposure"])
dataset = maker.run(empty_dataset, observation)

# Define sky model to simulate the data
spatial_model = GaussianSpatialModel(
    lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
)

spectral_model = ExpCutoffPowerLawSpectralModel(
    index=2,
    amplitude="3e-12 cm-2 s-1 TeV-1",
    reference="1 TeV",
    lambda_="0.05 TeV-1",
)


sky_model_simu = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="source"
)

bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
models = Models([sky_model_simu, bkg_model])
models_true = models.copy()  # comparison later between true and fitted values

print(models)

dataset.models = models
dataset.fake()

dataset.counts.sum_over_axes().plot(add_cbar=True);

# If you want to fit the data for comparison with MCMC later
# fit = Fit(dataset)
# result = fit.run(optimize_opts={"print_level": 1})


######################################################################
# Estimate parameter correlations with MCMC
# -----------------------------------------
# 
# Now let’s analyse the simulated data. Here we just fit it again with the
# same model we had before as a starting point. The data that would be
# needed are the following: - counts cube, psf cube, exposure cube and
# background model
# 
# Luckily all those maps are already in the Dataset object.
# 
# We will need to define a Likelihood function and define priors on
# parameters. Here we will assume a uniform prior reading the min, max
# parameters from the sky model.
# 


######################################################################
# Define priors
# ~~~~~~~~~~~~~
# 
# This steps is a bit manual for the moment until we find a better API to
# define priors. Note the you **need** to define priors for each parameter
# otherwise your walkers can explore uncharted territories (e.g. negative
# norms).
# 

print(dataset)

def init_model():

    # Define the free parameters and min, max values
    parameters = dataset.models.parameters

    # Setting the free/frozen parameters
    parameters["norm"].frozen = False

    parameters["sigma"].frozen = True
    parameters["lon_0"].frozen = True
    parameters["lat_0"].frozen = True
    parameters["tilt"].frozen = True

    # Setting the priors
    parameters["index"].prior = GaussianPrior(mu=2.0, sigma=0.5)
    parameters["norm"].prior = GaussianPrior(mu=1.0, sigma=0.1)

    # For uniform priors, choose how strong you want the prior to be
    weight = 10

    parameters["lambda_"].prior = UniformPrior(min=1e-2, max=1, weight=weight)
    parameters["amplitude"].prior = UniformPrior(
        min=3e-13, max=3e-11, weight=weight
    )
    parameters["sigma"].prior = UniformPrior(min=0.01, max=0.5, weight=weight)

    # Setting amplitude init values a bit offset to see evolution
    # Here starting close to the real value
    parameters["index"].value = 1.5
    parameters["amplitude"].value = 5e-12
    parameters["lambda_"].value = 0.5
    parameters["norm"].value = 0.9


init_model()

print(dataset.models)
print("stat =", dataset.stat_sum())

def lnprob(pars, dataset):
    """
    Estimate the likelihood of a model including prior on parameters.
    Input :
    pars : a list of parameters
    dataset: a gammapy dataset
    """
    # The MCMC sampler will evaluate the likelihood of the model given
    # a set of parameters. We need to update the model parameters before
    # evaluating the new likelihood value.
    for value, parameter in zip(
        pars, dataset.models.parameters.free_parameters
    ):
        parameter.value = value

    # dataset.stat_sum returns Cash statistics values that is minimized
    # emcee will maximisise the LogLikelihood so we need -dataset.stat_sum
    total_lnprob = (
        -dataset.stat_sum()
    )  # stat_sum now includes stat + stat_priors

    return total_lnprob

import emcee
import logging


nwalkers = 8
nrun = 2000

init_model()

p0 = [free_par.value for free_par in dataset.models.parameters.free_parameters]
labels = [
    free_par.name for free_par in dataset.models.parameters.free_parameters
]
ndim = len(p0)

rng = np.random.default_rng(seed=42)

randomize_walkers = rng.normal(1, 0.03, size=(nwalkers, len(p0)))
p0_walkers = (
    np.tile(p0, [nwalkers, 1]) * randomize_walkers
)  # init value for all walkers with slightly different values

print(dataset.models.parameters.free_parameters["amplitude"])
print(dataset.models.parameters.free_parameters["lambda_"])

print("Initial values for walkers are : ", p0_walkers)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    lnprob,
    args=[dataset],
)

log.info(f"Free parameters: {labels}")
log.info(f"Starting emcee sampling: nwalkers={nwalkers}, nrun={nrun}")


# Depending on your number of walkers, Nrun and dimensionality, this can take a while (> minutes)
state = sampler.run_mcmc(
    p0_walkers, nsteps=nrun, progress=True
)  # to speedup the notebook

samples1 = sampler.get_chain()

import zeus

nwalkers = 8
nrun = 1000

init_model()

p0 = [free_par.value for free_par in dataset.models.parameters.free_parameters]

# Use the same starting points for both methods
p0_walkers = (
    np.tile(p0, [nwalkers, 1]) * randomize_walkers
)  # init value for all walkers with slightly different values

print("Initial values for walkers are : ", p0_walkers)


sampler2 = zeus.EnsembleSampler(nwalkers, ndim, lnprob, args=[dataset])

log.info(f"Free parameters: {labels}")
log.info(f"Starting Zeus MCMC sampling: nwalkers={nwalkers}, nrun={nrun}")

# Depending on your number of walkers, Nrun and dimensionality, this can take a while (> minutes)
state = sampler2.run_mcmc(p0_walkers, nsteps=nrun, progress=True)
samples2 = sampler2.get_chain()


######################################################################
# Plot the results
# ----------------
# 
# The MCMC will return a sampler object containing the trace of all
# walkers. The most important part is the chain attribute which is an
# array of shape: *(nwalkers, nrun, nfreeparam)*
# 
# The chain is then used to plot the trace of the walkers and estimate the
# burnin period (the time for the walkers to reach a stationary stage).
# 

fig, axes = plt.subplots(len(labels), sharex=True, figsize=(10, 7))

for idx, ax in enumerate(axes):
    ax.plot(samples1[:, :, idx], "-k", alpha=0.2)  # emcee
    ax.plot(samples2[:, :, idx], "-b", alpha=0.2)  # Zeus MCMC
    ax.set_ylabel(labels[idx])

plt.xlabel("Nrun")
plt.show()


######################################################################
# Comparison of both algorithms
# =============================
# 
# | Note that the convergence is quite different between both MCMC
#   algorithms.
# | `zeus-mcmc` was able to converge to a steady solution much faster
#   than `emcee`. This means that you will burn less walkers steps and
#   in the end you will have a better sampling of your posterior
#   distributions. While `emcee` was faster to run you’ll have to
#   discard a larger fraction of the steps.
# 

from corner import corner

nburn1 = 800
nburn2 = 150

print("Corner plot with emcee")
s = samples1[nburn1:, :, :].reshape((-1, len(labels)))
corner(s, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.show()

print("Corner plot with Zeus MCMC")
s = samples2[nburn2:, :, :].reshape((-1, len(labels)))
corner(s, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.show()


######################################################################
# Plot the model dispersion
# -------------------------
# 
# | Using the samples from the chain after the burn period, we can plot
#   the different models compared to the truth model.
# | To do this we need to generate a spectral model for each parameter
#   state in the sample.
# | The shaded area will represent the uncertainty band.
# 

emin, emax = [0.1, 100] * u.TeV
nburn = 100
nmodel = 100  # number of samples to draw

samples = samples2

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for nwalk in range(0, nwalkers):
    for n in range(nburn, nburn + nmodel):
        pars = samples[n, nwalk, :]

        # set model parameters
        for i, free_par in enumerate(
            dataset.models.parameters.free_parameters
        ):
            free_par.value = pars[i]
        spectral_model = dataset.models["source"].spectral_model

        spectral_model.plot(
            energy_bounds=(emin, emax),
            ax=ax,
            energy_power=2,
            alpha=0.02,
            color="grey",
        )


sky_model_simu.spectral_model.plot(
    energy_bounds=(emin, emax), energy_power=2, ax=ax, color="red"
)
plt.show()


######################################################################
# Fun Zone
# --------
# 
# Now that you have the sampler chain, you have in your hands the entire
# history of each walkers in the N-Dimensional parameter space. You can
# for example trace the steps of each walker in any parameter space.
# 

# Here we plot the trace of one walker in a given parameter space
walkerid = 0
parx = 0
# Re-init the model to compare with the initial simulated parameters

free_pars = dataset.models.parameters.free_parameters
names = free_pars.names
true_pars = models_true.parameters

for i, name in enumerate(names):

    plt.plot(
        samples1[:, walkerid, parx],
        samples1[:, walkerid, i],
        ls=":",
        color="k",
        ms=1,
        label="emcee",
    )
    plt.plot(
        samples2[:, walkerid, parx],
        samples2[:, walkerid, i],
        ls=":",
        color="blue",
        ms=1,
        alpha=0.5,
        label="Zeus",
    )
    plt.plot(
        true_pars[parx].value,
        true_pars[name].value,
        "+",
        color="red",
        markersize=15,
        label="True value",
    )
    plt.xlabel(names[parx])
    plt.ylabel(name)
    plt.legend()
    plt.show()


######################################################################
# PeVatrons in CTA ?
# ==================
# 


######################################################################
# Now it’s your turn to play with this MCMC notebook. For example test the
# CTA performance to measure a cutoff at very high energies (100 TeV ?).
# 

