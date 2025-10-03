"""
Generating synthetic lightcurves and fitting the power spectral density of a lightcurve
=======================================================================================

This notebook presents the advanced Emmanoulopoulos algorithm for the
simulation of synthetic lightcurves. The original paper describing the
algorithm is linked `here <https://arxiv.org/pdf/1305.0304.pdf>`__. The
version implemented here is compatible with the Gammapy implementation
of the Timmer-Koenig algorithm. The Timmer-Koenig algorithm generates
synthetic lightcurve from a chosen power spectral density (PSD) shape.
However, it can only generate time series with a gaussian probability
density function (PDF). This is adequate for high-statistics
astrophysical domains such as the optical or X-rays, but can be in issue
when trying to reproduce curves in the gamma-ray domain, where photon
counts are lower and statistics are generally Poissonian. The
Emmanoulopoulos algorithm tries to solve this issue, combining a
requested PSD and PDF in the simulation. It provides accurate synthetic
lightcurves in a range of spectral indexes between -1 and -2 for
power-law or similar PSDs.

Together with the simulation algorithm the notebook shows a function to
compute the PSD envelope for a lightcurve using either the Timmer-Koenig
or the Emmanoulopoulos algorithm. This envelope is then used to fit the
PSD fot he observed lightcurve, by passing through a tailored
chi-squared-like cost function. This complex fitting is necessary to
account for the fact that the periodogram of the observed lightcurve is
only a possible realization of the PSD model, moreover convoluted with
Poissonian noise and instrumental responses. This can lead to biases or
deformation due to random fluctuation of the realization if extracted
with a simple curve fit of the periodogram.

The results are satisfactory for power-law or broken-power-law PSDs in a
physical interval of spectral indexes, between -1 and -2. Using the
Emmanoulopoulos algorithm shows slightly better PSD reconstruction over
the Timmer-Koenig after folding of the instrumental effects, at the cost
of a longer computation time. In general, using the Timmer-Koenig
algorithm is enough if the user is interested only in the frequency
domain, while using the Emmanoulopoulos becomes indespensable to model
correctly also the time domain.

The functions for the Timmer-Koenig and Emmanoulopoulos algorithms, the
envelope, x2-like cost function for fitting, and some helper analytical
functions are implemented in the helper package `gammapy_SyLC`.

"""


######################################################################
# Imports
# -------
# 
# The first step is importing some usual packages, needed Astropy
# utilities, scipy tools for PDFs and minimization, and Gammapy functions
# and classes for the observational part.
# 

import gammapy

print(f"Gammapy version : {gammapy.__version__}")

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import inspect

import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz

from regions import PointSkyRegion

from gammapy.estimators import LightCurveEstimator, FluxPoints
from gammapy.makers import SpectrumDatasetMaker
from gammapy.data import Observation, observatory_locations, FixedPointingInfo
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.irf import load_irf_dict_from_file
from gammapy.maps import MapAxis, RegionGeom, TimeMapAxis, RegionNDMap
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    LightCurveTemplateTemporalModel,
)
from gammapy.estimators.utils import compute_lightcurve_fvar
from gammapy.utils.random import get_random_state

from scipy.optimize import minimize
from scipy.signal import periodogram
from scipy.stats import lognorm


######################################################################
# Additionally, we import the necessary functions for lightcurve
# simulation and fitting from the helper package
# `gammapy_SyLC <https://github.com/cgalelli/gammapy_SyLC>`__. The package
# can be obtained from github and installed by running:
# 
# | `git clone https://github.com/cgalelli/gammapy_SyLC.git`
# | `cd gammapy_SyLC`
# | `python -m pip install -e .`
# 

from gammapy_SyLC import (
    pl, 
    lognormal,
    TimmerKonig_lightcurve_simulator, 
    Emmanoulopoulos_lightcurve_simulator, 
    lightcurve_psd_envelope,
    psd_fit,
)


######################################################################
# Reference Lightcurve
# --------------------
# 
# As a reference, the notebook uses the H.E.S.S. dataset for the PKS2155
# AGN flare of 2006. Data properties such as mean and standard deviation
# fo the norm, number of points, sampling frequency, are taken from this
# flare. The synthetic lightcurve will be oversampled by a factor 10.
# 

lc_path = Path("$GAMMAPY_DATA/estimators/")
lc_filename = "pks2155_hess_lc/pks2155_hess_lc.fits"

lc = FluxPoints.read(lc_path / lc_filename, format="lightcurve")
odata = lc.norm.data.flatten()
omean = odata.mean()
ostd = odata.std()
npoints = len(lc.norm.data) * 10
times = lc.geom.axes["time"].edges
tref = lc.geom.axes["time"].reference_time
smax = np.diff(times).max() / 10
lc.plot()

plt.show()


######################################################################
# Simulation
# ----------
# 
# As a first step, the parameters for the functions used as models in the
# simulations are setup here:
# 

ln_params = {"s": 0.5}
pl_params = {"index": -1.4}


######################################################################
# Both the TK and EMM algorithms are called with the same power-law PSD.
# The EMM algorithm uses a lognormal PDF. The difference between TK and
# EMM algorithms is shown in the leftmost and rightmost plot, where the
# gaussian vs lognormal shape is evident. The middle plot shows the
# perfect compatibility in the periodogram. Seed is fixed for
# reproducibility.
# 

# %%time
seed = 532019
# seed = "random-seed"
lctk2, taxis2 = Emmanoulopoulos_lightcurve_simulator(
    lognormal,
    pl,
    npoints,
    smax,
    pdf_params=ln_params,
    psd_params=pl_params,
    mean=omean,
    std=ostd,
    random_state=seed,
)
lctk, taxis = TimmerKonig_lightcurve_simulator(
    pl,
    npoints,
    smax,
    power_spectrum_params=pl_params,
    mean=omean,
    std=ostd,
    random_state=seed,
)
freqstk, pgramtk = periodogram(lctk, 1 / smax.value)
freqstk2, pgramtk2 = periodogram(lctk2, 1 / smax.value)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
ax1.plot(taxis, lctk)
ax2.loglog(freqstk[1:], pgramtk[1:])
ax3.hist(lctk)
ax1.plot(taxis2, lctk2)
ax2.loglog(freqstk2[1:], pgramtk2[1:])
ax3.hist(lctk2, alpha=0.5)


ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Arb. Units')
ax2.set_xlabel('Frequency [1/s]')
ax2.set_ylabel('PSD')
ax3.set_xlabel('Arb. Units')
ax3.set_ylabel('Counts')


coeff = np.polyfit(np.log(freqstk[1:]), np.log(pgramtk[1:]), 1)
coeff2 = np.polyfit(np.log(freqstk2[1:]), np.log(pgramtk2[1:]), 1)

print(coeff, coeff2)


######################################################################
# Gammapy setup and simulation
# ----------------------------
# 
# Setup of geometry for the Gammapy simulation. Generic setup for
# pointing, energy binning, and IRFs. For realistic simulations, choose
# IRFs that are consistent with the instrument and observational
# conditions.
# 

TimeMapAxis.time_format = "iso"

path = Path("$GAMMAPY_DATA/cta-caldb")
irf_filename = "Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"

irfs = load_irf_dict_from_file(path / irf_filename)

energy_axis = MapAxis.from_energy_bounds(
    energy_min=0.1 * u.TeV, energy_max=100 * u.TeV, nbin=1
)

energy_axis_true = MapAxis.from_edges(
    np.logspace(-1.2, 2.0, 31), unit="TeV", name="energy_true", interp="log"
)

time_axis = MapAxis.from_nodes(taxis, name="time", interp="lin")

geom = RegionGeom.create(
    "galactic;circle(107.65, -40.17, 5)", axes=[energy_axis]
)

pointing_position = SkyCoord(107.65, -40.17, unit="deg", frame="galactic")
pointing = FixedPointingInfo(
    fixed_icrs=SkyCoord(107.65, -40.17, unit="deg", frame="galactic").icrs,
)


######################################################################
# The time series generated via EMM is taken as a
# LightCurveTemplateTemporalModel
# 

gti_t0 = tref

spectral_model = PowerLawSpectralModel(
    amplitude=1e-10 * u.TeV**-1 * u.cm**-2 * u.s**-1
)

m = RegionNDMap.create(
    region=PointSkyRegion(center=pointing_position),
    axes=[time_axis],
    unit="cm-2s-1TeV-1",
)

m.quantity = lctk2

temporal_model = LightCurveTemplateTemporalModel(m, t_ref=gti_t0)

model_simu = SkyModel(
    spectral_model=spectral_model,
    temporal_model=temporal_model,
    name="model-simu",
)


######################################################################
# Observation timing setup and simulation fo the datasets. The
# “observational” sampling is taken to be much sparser than the synthetic
# lightcurve, to avoid aliasing.
# 

lvtm = 10 * u.min
tstart = gti_t0 + np.arange(npoints / 10) * lvtm
altaz = pointing_position.transform_to(
    AltAz(obstime=tstart, location=observatory_locations["cta_south"])
)

datasets = Datasets()

empty = SpectrumDataset.create(
    geom=geom, energy_axis_true=energy_axis_true, name="empty"
)

maker = SpectrumDatasetMaker(selection=["exposure", "background", "edisp"])

for idx in range(len(tstart)):
    obs = Observation.create(
        pointing=pointing,
        livetime=lvtm,
        tstart=tstart[idx],
        irfs=irfs,
        reference_time=gti_t0,
        obs_id=idx,
        location=observatory_locations["cta_south"],
    )
    empty_i = empty.copy(name=f"dataset-{idx}")
    dataset = maker.run(empty_i, obs)
    dataset.models = model_simu
    dataset.fake()
    datasets.append(dataset)


spectral_model = PowerLawSpectralModel(
    amplitude=7e-11 * u.TeV**-1 * u.cm**-2 * u.s**-1
)
model_fit = SkyModel(spectral_model=spectral_model, name="model-fit")
datasets.models = model_fit

datasets.gti.table


######################################################################
# Lightcurve estimator setup and run.
# 

lc_maker_1d = LightCurveEstimator(
    energy_edges=[0.1, 100] * u.TeV,
    source="model-fit",
    selection_optional=["ul"],
)

lc_1d = lc_maker_1d.run(datasets)
lc_1d.plot();


######################################################################
# Assessment of the properties of the “observed” lightcurve in the time
# and frequency domain.
# 

data = lc_1d.norm.data.flatten()
dmean = data.mean()
dstd = data.std()
dnpoints = len(data)
dtimes = lc_1d.geom.axes["time"].edges
dsmax = np.diff(dtimes).max()
ffreqs, pgram = periodogram(data, 1 / dsmax.value)
coeff = np.polyfit(np.log(ffreqs[1:]), np.log(pgram[1:]), 1)
print(f"Index from simple line fit: {coeff[0]}")
plt.loglog(ffreqs[1:], pgram[1:])


######################################################################
# Fitting
# ~~~~~~~
# 
# The psd_fit function calls the internal x2_fit as a cost function for
# the scipy minimizer, providing a fit of the spectral index for the
# “observed” lightcurve, in this case assuming a power-law PSD. Since here
# only the PSD is fit, and the fluxes are relatively high, the TK
# algorithm can be used to save time.
# 

# %%time
index, index_err = psd_fit(pgram,
                           pl, 
                           {"index":-1.4}, 
                           dsmax, 
                           simulator="TK", 
                           nsims=5000, 
                           mean=dmean, 
                           std=dstd,
                           nexp=100,
                          )

print(f"Fit index: {index[0]}\u00B1{index_err[0]}")

envelopes, freqs = lightcurve_psd_envelope(
    pl,
    dnpoints,
    dsmax,
    psd_params={"index": index},
    simulator="TK",
    nsims=5000,
    mean=dmean,
    std=dstd,
)
qmin2, qmin1, qmax1, qmax2 = np.quantile(envelopes, [0.025, 0.16, 0.84, 0.975], axis=0)
plt.plot(freqs, np.median(envelopes, axis=0))
plt.fill_between(freqs, qmin2, qmax2, alpha=0.3, color='#1f77b4')
plt.fill_between(freqs, qmin1, qmax1, alpha=0.2, color='#1f77b4')

plt.plot(freqs, pgram[1:], linewidth=0.7, marker="d")
plt.yscale("log")
plt.show()


######################################################################
# Recipe by `Claudio Galelli <https://github.com/cgalelli/>`__
# 