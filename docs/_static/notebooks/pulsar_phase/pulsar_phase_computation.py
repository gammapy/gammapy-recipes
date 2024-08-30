#!/usr/bin/env python
# coding: utf-8

# # Phase computation for pulsar using PINT

# This notebook has been done for the following version of Gammapy and PINT:
# 
# Gammapy version : 1.2
# 
# PINT version : 1.0

# This notebook shows how to compute and add the phase information into the events files of pulsar observations. This step is needed to perform the pulsar analysis with Gammapy and should be the first step in the high level analysis. For the pulsar analysis we need two ingredients:
# 
# 1. The time of arrivals (TOAs). These times should have very high precision due to the common fast periods of pulsars. Usually these times are already stored in the EventList. For the computation of pulsar timing, times must be corrected in order to be referenced in the Solar System barycenter (SSB) because this system can nearly be regarded as an inertial reference frame with respect to the pulsar.
# 
# 
# 2. The model of rotation of the pulsar, also known as ephemeris, at the epoch of the observations. These ephemerides are stored in a specific format and saved as .par files which contain the periods, derivatives of the periods, coordinates, glitches, etc.
# 
# __For the following steps of this tutorial, we need the original EventLists from the DL3 files, and a model in .par format.__
# 
# The main software that we will use to make the barycentric corrections and the phase-folding to the model is the PINT python library, [Luo J., Ransom S. et al., 2021](https://arxiv.org/abs/2012.00074), [ASCL](http://ascl.net/1902.007).
# For more information about this package, see [PINT documentation](https://nanograv-pint.readthedocs.io/en/latest/). 

# ## 0. Dependencies and imports

# To run this notebook, you must have Gammapy and PINT (see documentation above) installed in the same environment. We recommend installing Gammapy first and then installing PINT using your preferred package manager.
# 
# 
# `$ conda env create -n gammapy-pint -f gammapy-pint-environment.yml`
# 
# `$ conda activate gammapy-pint`
# 
# `$ pip install pint-pulsar`

# Alternatively, one can also run the yaml environement file provided in the folder of this notebook:
# 
# `$ conda env create -n gammapy-pint -f gammapy-pint-environment.yml`
# 

# In[1]:


import gammapy
import pint

print(f"Gammapy version : {gammapy.__version__}")
print(f"PINT version : {pint.__version__}")


# In[2]:


import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np
from pathlib import Path
from gammapy.data import DataStore, EventList, Observation
import logging

log = logging.getLogger(__name__)


# We also need some imports from PINT:

# In[3]:


import pint.models as pmodels
from pint import toa


# ## 1. Reading DataStore

# First we need to define the data sample. In this notebook we will use two runs from the MAGIC gammapy data sample available in https://github.com/gammapy/gammapy-data

# In[4]:


# Define the directory containing the DL3 data
DL3_dir = "$GAMMAPY_DATA/magic/rad_max/data"


# In[5]:


# Read DataStore from a directory
data_store = DataStore.from_dir(DL3_dir)


# Let's run this tutorial for the Crab pulsar :

# In[6]:


target_pos = SkyCoord(ra=083.633, dec=+22.014, unit="deg", frame="icrs")


# In[7]:


selection = dict(
    type="sky_circle",
    frame="icrs",
    lon=target_pos.ra,
    lat=target_pos.dec,
    radius="5 deg",
)
selected_obs_table = data_store.obs_table.select_observations(selection)


# In[8]:


obs_id = selected_obs_table["OBS_ID"]
print(obs_id)


# For the following we will take the run 5029747.

# In[9]:


observations = data_store.get_observations(
    [5029747], required_irf="point-like"
)


# In[10]:


print(observations)


# ## 2. Phase-folding with PINT for one observation

# Let's extract the times from the observation:

# In[11]:


# Extract times from EventList
observation = observations[0]
times = observation.events.time


# In[12]:


print(times)


# Now we have the TOAs for the events in the system of the telescope. Please note: the actual precision of the times is higher than the displayed output (and we really need this precision for the pulsar analysis!). In the next step, the timing in the SSB and the phase for each TOA has to be created.

# ## 2.1 An ephemeris file from Fermi-LAT data.

# In order to compute the phases of a pulsar, one needs an ephemeris file, typically stored as a .par file.
# 
# In the following, we will use an ephemeris file for the Crab provided by Fermi-LAT, see [Kerr, M.; Ray, P. S.; et al; 2015](https://arxiv.org/abs/1510.05099). This ephemeris file for the Crab pulsar can be found alongside other pulsar ephemeris files at this [confluence page]( https://confluence.slac.stanford.edu/display/GLAMCOG/LAT+Gamma-ray+Pulsar+Timing+Models). 
# 
# However, it is important to note that many of the ephemeris files are not up-to-date. Therefore, they could give bad results on the phase computation. In particular, you should always check that the MJD of the observations one wants to phase lies between the `START` and `FINISH` entries of the ephemeris file (see next section).

# In[13]:


# Path to the ephemeris file
ephemeris_file = "0534+2200_ApJ_708_1254_2010.par"


# Note that *Fermi*-LAT ephemeris files are created primarily by and for [Tempo2](https://www.pulsarastronomy.net/pulsar/software/tempo2). Most of the time, using such ephemeris file with PINT will not raise any issues. However, in a few cases, PINT does not support features from Tempo2. 
# 
# In our case, an error occurs when using the ephemeris file with PINT. This is due to the `JUMP` line. To proceed, simply comment  out the line (with #) or remove it. Note that this line is not important for the gamma-ray instruments, so it is acceptable to disregard it.

# ## 2.2 Computing pulsar phases

# Now that we have the model and the times of arrival for the different events, we can compute the timing corrections and the pulsar phases needed for the pulsar analysis. In this case, we use the PINT package described in the introduction.

# First we will explore our model. We print some of the relevant quantities

# In[14]:


model = pmodels.get_model(ephemeris_file)
print(model.components["AstrometryEquatorial"])
print(model.components["SolarSystemShapiro"])
print(model.components["DispersionDM"])
print(model.components["AbsPhase"])
print(model.components["Spindown"])


# There are multiple parameters such as the name of the source, the frequencies of rotation and its derivatives (F0,F1,F2), the dispersion measure, etc. Check the [PINT documentation](https://nanograv-pint.readthedocs.io) for a list of additional parameters. To obtain the complete set of parameters from the ephemeris file, one can simply print the model:
# `print(model)`

# As mentioned previously, we should ensure the time of the observation lies within the ephemeris time definition. In our example, we only have one run, so we can check that manually:

# In[15]:


print(
    f"Ephemeris time definition:\n{model.START.value} - {model.FINISH.value}"
)
print(
    f"Observation time definition:\n{observation.tstart} - {observation.tstop}"
)


# If you have several observations that are sorted by time, you can manually check for the start time of the first observation and the stop time of the last one. Otherwise, you can create a small function like the following one:

# In[16]:


def check_time(observation, timing_model):
    """
    Check that the observation time lies within the time definition of the pulsar
    timing model.

    Parameters
    ----------
    observation: `gammapy.data.Observation`
        Observation to check.
    timing_model: `pint.models.TimingModel`
        The timing model that will be used.
    """
    model_time = Time(
        [model.START.value, model.FINISH.value], scale="tt", format="mjd"
    )
    if (model_time[0].value > observation.tstart.tt.mjd) or (
        model_time[1].value < observation.tstop.tt.mjd
    ):
        log.warning(
            f"Warning: Observation time of observation {observation.obs_id} goes out of timing model validity time."
        )


# In[17]:


check_time(observation, model)


# Now we can compute the phases. For that, we define a list of TOA objects that are the main object of PINT.

# In[18]:


get_ipython().run_cell_magic('time', '', '\n# Set these to True is your observatory has clock correction files.\n# If it is set to True but your observatory does not have clock correction files, it will be ignored.\ninclude_bipm = False\ninclude_gps = False\n\n# Set this to True or False depending on your ephemeris file.\n# Here we can see that the \'PLANET_SHAPIRO\' entry is \'N\' so we set it to True.\nplanets = False\n\n# Create a TOA object for each time\ntoas = toa.get_TOAs_array(\n    times=times,\n    obs="magic",\n    errors=1 * u.microsecond,\n    ephem="DE421",\n    include_gps=include_gps,\n    include_bipm=include_bipm,\n    planets=planets,\n)')


# Once we have the TOAs object and the model, the phases are easily computed using the model.phase() method. Note that the phases are computed in the interval [-0.5,0.5]. Most of the time, we use the phases in the interval [0,1] so we have to shift the negative ones.

# In[19]:


# Compute phases
phases = model.phase(toas, abs_phase=True)[1]

# Shift phases to the interval (0,1]
phases = np.where(phases < 0.0, phases + 1.0, phases)


# ## 3. Adding phases and metadata to an EventList and put it in a new Observation. 

# Once the phases are computed we need to create a new EventList table that includes both the original information of the events and the phase information in extra columns. This is necessary for Gammapy to read the phases and use them as an extra variable of each event.

# In[20]:


# Extract the table of the EventList
table = observation.events.table


# In[21]:


# Show original table
table


# In[22]:


# Add a column for the phases to the table
table["PHASE"] = phases.astype("float64")


# Note that you can add multiple columns to a same file, only the name of the column has to be unique, eg `table['PHASE_SRC1']`, `table['PHASE_SRC2']` etc"

# In[23]:


# Show table with phases
table


# Now we can see that the 'PHASE' column has been added to the table

# At this point, we also want to add metadata to the table. It is very useful to keep track of what has been done to the file. For instance, if a file contains multiple pulsars, we want identify quickly which column corresponds to each pulsar. Moreover, experience has shown that it is common to have different ephemeris files for the same pulsar. Therefore, it is useful to have several phase columns in the same file to easily identify which column corresponds to each ephemeris file, parameters, etc.
# 
# Since there is currently no "standard" format for such metadata, we propose a template for the essential information that one wants to save in the header of the event file. First, we look at the present meta info on the table.

# In[24]:


table.meta


# In[25]:


def get_log(ephemeris_file, phase_column_name="PHASE"):
    return (
        "COLUMN_PHASE: "
        + str(phase_column_name)
        + "; PINT_VERS: "
        + pint.__version__
        + "; GAMMAPY_VERS: "
        + gammapy.__version__
        + "; EPHEM_FILE: "
        + ephemeris_file
        + "; PSRJ :"
        + str(model.PSR.value)
        + "; START: "
        + str(model.START.value)
        + "; FINISH: "
        + str(model.FINISH.value)
        + "; TZRMJD: "
        + str(model.TZRMJD.value)
        + "; TZRSITE: "
        + str(model.TZRSITE.value)
        + "; TZRFREQ: "
        + str(model.TZRFRQ.value)
        + "; EPHEM: "
        + str(model.EPHEM.value)
        + "; EPHEM_RA: "
        + str(model.RAJ.value)
        + "; EPHEM_DEC: "
        + str(model.DECJ.value)
        + "; PHASE_OFFSET: "
        + "default = 0"
        + "; DATE: "
        + str(Time.now().mjd)
        + ";"
    )


# In[26]:


phase_log = get_log(ephemeris_file=ephemeris_file, phase_column_name="PHASE")
print(phase_log)


# In[27]:


# Add the generated string to the meta data of the table
table.meta["PH_LOG"] = phase_log


# In[28]:


table.meta


# Once this is done, we can put back the table in a new `EventList` object and in a new `Observation` object. 

# In[29]:


# Create new event list and add it to observation object
new_event_list = EventList(table)
new_obs = observation.copy(in_memory=True, events=new_event_list)


# In[30]:


new_obs.events.table


# ## 4. Save new Event List and writing a modify HDU index table

# In the following, we show how to write the files in a directory contained in the original datastore directory. This follows the logic of DL3 data store and facilitates the manipulation of the HDU table.
# 
# If you do not want to save the events files bur rather directly perform the pulsar analysis, you can skip both this step and the step of the handling metadata. However, be aware that for large datasets, the computation of the phases can take tens of minutes.

# In[31]:


data_store.hdu_table.base_dir


# In[32]:


# Define output directory and filename
datastore_dir = str(data_store.hdu_table.base_dir) + "/"
output_directory = "pulsar_events_file/"
output_path = datastore_dir + output_directory
filename = f"dl3_pulsar_{observation.obs_id:04d}.fits.gz"
file_path = output_path + filename

Path(output_path).mkdir(parents=True, exist_ok=True)


# In[33]:


output_path


# In[34]:


# Save the observation object in the specified file_path
print("Writing output file in " + str(file_path))
new_obs.write(path=file_path, include_irfs=False, overwrite=True)


# Once the file has been written, we want to write a modified version of the HDU table. This is mandatory if we want to open the phased events file together with its associated IRFs. 

# In[35]:


# Print the current data store HDU table.
new_hdu = data_store.hdu_table.copy()
new_hdu


# In[36]:


for entry in new_hdu:
    if (entry["HDU_NAME"] == "EVENTS") and (
        entry["OBS_ID"] == observation.obs_id
    ):
        entry["FILE_DIR"] = "./" + str(output_directory)
        entry["FILE_NAME"] = filename


# In[37]:


new_hdu


# We see that the `FILE_DIR`and `FILE_NAME`entry have been modified for our phased events file.

# Finally, we need to save the new HDU table in the original DL3 directory. One must be very careful with naming the new HDU file, such that it does not have the same name as the original HDU file of the data store. Otherwise, the original HDU file will be overwritten.

# In[38]:


new_hdu.write(
    datastore_dir + "hdu-index-pulsar.fits.gz", format="fits", overwrite=True
)


# **Note: Here we demonstrate only one approach that could be useful, showing the steps to save the new Event files in a directory and generate a new modified HDU index table. However, the user is free to choose the absolute path of the EventList and DataStore. Another approach, for instance, could be making a full copy of the DataStore, or changing the location of the pulsar event files to one that is more convenient for the user.**

# ## 5. Opening the new DataStore

# Once all of this is done, we just have to open the data store using DataStore.from_dir() and pass the pulsar HDU table to it :

# In[39]:


pulsar_datastore = DataStore.from_dir(
    DL3_dir, hdu_table_filename="hdu-index-pulsar.fits.gz"
)


# In[40]:


observations = pulsar_datastore.get_observations(
    [5029747], required_irf="point-like"
)
observations[0].available_hdus


# In[41]:


observations[0].events.table


# We can see that we recover both the IRFs and the events file with the phase column.

# ## 6. Pulsar analysis tools with gammapy

# Once we have the correct DataStore and the modified EventList with the phase information, we can perform the pulsar analysis using different tools available in Gammapy. Allowing us to compute the phaseogram, maps, SED, lightcurve and more. To do so, please refer to the following [Gammapy tutorial](https://docs.gammapy.org/1.0/tutorials/analysis-time/pulsar_analysis.html#sphx-glr-tutorials-analysis-time-pulsar-analysis-py).

# 
# Recipe made by [Alvaros Mas](https://github.com/alvmas), [Maxime Regeard](https://github.com/MRegeard), [Jan Lukas Schubert](https://github.com/jalu98).
