"""
TSMapEstimator vs. ExcessMapEstimator
=====================================

This notebooks compares the results of the full likelihood fitting
approach (“forward folding”) against the Li&Ma solution of the same
problem (“backwards folding”). The results are expected to be
approximately equivalent in case of using a disk kernel for the
`TSMapEstimator` and neclecting the effect of the PSF.

The remaining difference is that while the `ExcessMapEstimator` sums
all counts and computes flux based on the mean exposure in the same
region, the `TSMapEstimator` takes into account the full pixel by
pixel information.

Results: - The results are indeed approximately equivalent - On average
the `TSMapEstimator` find slightly higher errors because it takes into
account the fluctuations per pixel - The `ExcessMapEstimator` is
faster by a factor of 50-100 for ordinary flux and error estimation.
Once assymetric errors and ULs are computed the `TSMapEstimator` shows
the better performance and is ~2-3 times faster. When taking
multiprocessing into account it is even faster by a factor of ~10.

"""

from gammapy.estimators import ExcessMapEstimator, TSMapEstimator
from gammapy.datasets import MapDataset
from gammapy.modeling.models import (
    SkyModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    DiskSpatialModel,
)
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

dataset = MapDataset.read(
    "$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz", name="cta_dataset"
)
dataset.psf = None

est = ExcessMapEstimator(
    correlation_radius="0.1 deg",
)

model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    spatial_model=DiskSpatialModel(r_0="0.1 deg"),
)

ts_est = TSMapEstimator(
    model=model,
    sum_over_energy_groups=True,
)

est_all = ExcessMapEstimator(
    correlation_radius="0.1 deg", selection_optional="all"
)


ts_est_all = TSMapEstimator(
    model=model,
    sum_over_energy_groups=True,
    selection_optional="all",
    n_jobs=8,
)

# %%time
result = est.run(dataset=dataset)

# %%time
result_ts = ts_est.run(dataset)

# %%time
result = est_all.run(dataset=dataset)

# %%time
result_ts = ts_est_all.run(dataset)

def plot_side_by_side(m_1, m_2, **kwargs):
    fig = plt.figure(figsize=(14, 12))
    ax1 = plt.subplot(221, projection=m_1.geom.wcs)
    ax2 = plt.subplot(222, projection=m_1.geom.wcs)
    ax3 = plt.subplot(223, projection=m_1.geom.wcs)
    ax4 = plt.subplot(224)

    m_1.plot(ax1, **kwargs)
    ax1.set_title("ExcessMapEstimator")

    m_2.plot(ax2, **kwargs)
    ax2.set_title("TSMapEstimator")

    ax4.scatter(m_1.data.flat, m_2.data.flat, s=2, alpha=0.05)
    ax4.set_title("Correlation")
    ax4.set_xlabel("ExcessMapEstimator")
    ax4.set_ylabel("TSMapEstimator")
    ax4.set_xlim(kwargs["vmin"], kwargs["vmax"])
    ax4.set_ylim(kwargs["vmin"], kwargs["vmax"])
    x = np.linspace(kwargs["vmin"], kwargs["vmax"], 3)
    ax4.plot(x, x, color="k")
    ax4.set_aspect("equal")

    diff = m_1 - m_2
    vmax = -0.1 * kwargs["vmax"]
    diff.plot(
        ax=ax3,
        cmap="RdBu",
        stretch="linear",
        add_cbar=True,
        vmin=-vmax,
        vmax=vmax,
    )
    ax3.set_title("Difference (ExcessMap - TSMap)")


######################################################################
# Flux
# ----
# 

plot_side_by_side(
    result.flux, result_ts.flux, add_cbar=True, vmin=-0.3e-11, vmax=2e-11
);


######################################################################
# Flux Error
# ----------
# 

plot_side_by_side(
    result.flux_err,
    result_ts.flux_err,
    add_cbar=True,
    vmin=0.5e-12,
    vmax=2e-12,
);


######################################################################
# Flux UL
# -------
# 

plot_side_by_side(
    result.flux_ul, result_ts.flux_ul, add_cbar=True, vmin=-0.3e-11, vmax=2e-11
);


######################################################################
# Flux Error Pos.
# ---------------
# 

plot_side_by_side(
    result.flux_errp,
    result_ts.flux_errp,
    add_cbar=True,
    vmin=0.5e-12,
    vmax=2e-12,
);


######################################################################
# Flux Error Neg.
# ---------------
# 

# result._data["norm_errn"].data *= -1

plot_side_by_side(
    result.flux_errn,
    result_ts.flux_errn,
    add_cbar=True,
    vmin=0.4e-12,
    vmax=2e-12,
);


######################################################################
# Sqrt(TS)
# --------
# 

plot_side_by_side(
    result.sqrt_ts,
    result_ts.sqrt_ts,
    add_cbar=True,
    vmin=-5,
    vmax=20,
    stretch="linear",
);


######################################################################
# NPred Excess
# ------------
# 

plot_side_by_side(
    result.npred_excess,
    result_ts.npred_excess,
    add_cbar=True,
    vmin=-40,
    vmax=500,
    stretch="sqrt",
);


######################################################################
# NPred Ref.
# ----------
# 

plot_side_by_side(
    result.npred_excess_ref,
    result_ts.npred_excess_ref,
    add_cbar=True,
    vmin=0,
    vmax=500,
);


######################################################################
# NPred Err
# ---------
# 

plot_side_by_side(
    result.npred_excess_err,
    result_ts.npred_excess_err,
    add_cbar=True,
    vmin=0,
    vmax=30,
);


######################################################################
# NPred Error Pos.
# ----------------
# 

plot_side_by_side(
    result.npred_excess_errp,
    result_ts.npred_excess_errp,
    add_cbar=True,
    vmin=0,
    vmax=30,
);


######################################################################
# NPred Error Neg.
# ----------------
# 

plot_side_by_side(
    result.npred_excess_errn,
    result_ts.npred_excess_errn,
    add_cbar=True,
    vmin=0,
    vmax=30,
);


######################################################################
# NPred UL
# --------
# 

plot_side_by_side(
    result.npred_excess_ul,
    result_ts.npred_excess_ul,
    add_cbar=True,
    vmin=-30,
    vmax=500,
    stretch="sqrt",
);