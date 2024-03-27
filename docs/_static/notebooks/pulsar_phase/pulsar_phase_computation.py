#!/usr/bin/env python
# coding: utf-8

# # Phase computation for pulsar using PINT

# This notebook has been done for the following version of Gammapy and PINT:
# 
# Gammapy version : 1.0.1
# 
# PINT version : 0.9.5

# This notebook shows how to compute and add the phase information into the events files of pulsar observations. This step is needed to perform the pulsar analysis with Gammapy and should be the first step in the high level analysis. For the pulsar analysis we need two ingredients:
# 
# 1. The time of arrivals (TOAs). These times should have very high precision due to the common fast periods of pulsars. Usually these times are already stored in the EventList. For the computation of pulsar timing, times must be corrected in order to be referenced in the Solar System barycenter (SSB) because this system can nearly be regarded as an inertial reference frame with respect to the pulsar.
# 
# 
# 2. The model of rotation of the pulsar, also known as ephemeris, at the epoch of the observations. These ephemerides are stored in an specific format and saved as .par files and contain informations on the periods, derivatives of the periods, coordinates, glitches, etc.
# 
# __For the following steps of this tutorial, we need the original EventLists from the DL3 files, and a model in .par format.__
# 
# The main software that we will use to make the barycentric corrections and the phase-folding to the model is the PINT python library, [Luo J., Ransom S. et al., 2021](https://arxiv.org/abs/2012.00074), [ASCL](http://ascl.net/1902.007).
# For more information about this package, see [PINT documentation](https://nanograv-pint.readthedocs.io/en/latest/). 

# ## 0. Dependencies and imports

# In order to run this notebook, one needs to have installed Gammapy as well as PINT (see documentation above) in the same environment. We recommend to first install Gammapy and then install PINT using your prefered package manager.
# 
# 
# `$ conda env create -n gammapy-pint -f gammapy-1.0-environment.yml`
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


# And we also need some imports from PINT:

# In[3]:


import pint.models as pmodels
from pint import toa


# ## 1. Reading DataStore

# First we neeed to define the data sample. In this notebook we will use two runs from the MAGIC gammapy data sample available in https://github.com/gammapy/gammapy-data

# In[4]:


# Define the directory containing the DL3 data
DL3_direc = "$GAMMAPY_DATA/magic/rad_max/data"


# In[5]:


# Read DataStore from a directory
data_store = DataStore.from_dir(DL3_direc)


# Let's run this tutorial for the Crab pulsar :

# In[6]:


target_pos = SkyCoord(
    ra=083.6331144560900, dec=+22.0144871383400, unit="deg", frame="icrs"
)


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


# Now we have the TOAs of the events in the system of the telescope. Please note that the actual precision of the times is higher than the diplayed output (and we really need this precision for the pulsar analysis!). In the next step, the timing in the SSB and the phase for each TOA has to be created. 

# ## 2.1 An ephemeris file from Fermi-LAT data.

# In order to compute the phases of a pulsar, one needs an ephemeris file, usually store as a .par file. 
# 
# In the following, we will use an ephemeris file for the Crab provided by Fermi-LAT, see [Kerr, M.; Ray, P. S.; et al; 2015](https://arxiv.org/abs/1510.05099). This ephemeris file for the Crab pulsar can be found alongside other pulsar ephemeris files at this [confluence page]( https://confluence.slac.stanford.edu/display/GLAMCOG/LAT+Gamma-ray+Pulsar+Timing+Models). 
# 
# However, be aware that most of these ephemeris files are not up-to-date. Therefore they could give bad results on the phase computation. In particular, one should always checked that the MJD of the observations one wants to phased lies between the `START`and `FINISH`entry of the ephemeris file.

# In[13]:


# Path to the ephemeris file
ephemeris_file = "./0534+2200_ApJ_708_1254_2010.par"


# Note that sometimes one needs to change some of the parameters of the ephemeris file that are not used in gamma-ray astronomy by hand. For instance, here we have removed the 'JUMP' line since it does not have any effect in our computation and raise an error in PINT. The ephemeris file provided with this notebook does not have this line. 

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


# There are multiple parameters such as the name of the source, the interval of validity of the model (START to FINISH), the frequencies of rotation and its derivatives (F0,F1,F2). There are other additional parameters that can be checked in the [PINT documentation](https://nanograv-pint.readthedocs.io)

# Now we can compute the phases. For that, we define a list of TOA objects that are the main object of PINT.

# In[15]:


get_ipython().run_cell_magic('time', '', '\n# Put these to True is your observatory has clock correction files.\n# If it is set to True but your observatory does not have clock correction files, it will be ignored.\ninclude_bipm = False\ninclude_gps = False\n\n# Set this to True or False depending on your ephemeris file.\n# Here we can see that the \'PLANET_SHAPIRO\' entry is \'N\' so we set it to True.\nplanets = False\n\n# Create a TOA object for each time\ntoas = toa.get_TOAs_array(\n    times=times,\n    obs="magic",\n    errors=1 * u.microsecond,\n    ephem="DE421",\n    include_gps=include_gps,\n    include_bipm=include_bipm,\n    planets=planets,\n)')


# Once we have the TOAs object and the model, the phases are easily computed using the model.phase() method. Note that the phases are computed in the interval [-0.5,0.5]. Most of the times, we use the phases in the interval [0,1] so we have to shift the negative ones.

# In[16]:


# Compute phases
phases = model.phase(toas, abs_phase=True)[1]

# Shift phases to the interval (0,1]
phases = np.where(phases < 0.0, phases + 1.0, phases)


# ## 3. Adding phases and metadata to an EventList and put it in a new Observation. 

# Once the phases are computed we need to create a new EventList table that includes both the original information of the events and the phase information in extra columns. This is necessary for Gammapy to read the phases and use them as an extra variable of each event.

# In[17]:


# Extract the table of the EventList
table = observation.events.table


# In[18]:


# Show original table
print(table)


# In[19]:


# Add a column for the phases to the table
table["PHASE"] = phases.astype("float64")


# Note that you can add multiple columns to a same file, only the name of the column has to be unique, eg `table['PHASE_SRC1']`, `table['PHASE_SRC2']` etc"

# In[20]:


# Show table with phases
table


# Now we can see that the 'PHASE' column has been added to the table

# At this point, we also want to add meta data to the table. It is very useful to keep track of what has been done to the file. For instance, if we have multiple pulsars in the same file, we want to be able to know quickly which column correspond to which pulsar. Moreover, experience shows that one often use different ephemeris file for the same pulsar. Therefore, it is very useful to have several phase columns in the same file and to be able to know which column correspond to which ephemeris file, parameters, etc.
# 
# Since there is not yet a "standard" format for such metadata, we propose a template for the essential informations that one wants to save in the header of the event file. First, we look at the present meta info on the table.

# In[21]:


table.meta


# In[22]:


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


# In[23]:


phase_log = get_log(ephemeris_file=ephemeris_file, phase_column_name="PHASE")
print(phase_log)


# In[24]:


# Add the generated string to the meta data of the table
table.meta["PH_LOG"] = phase_log


# In[25]:


table.meta


# Once this is done, we can put back the table in a new `EventList` object and in a new `Observation` object. 

# In[26]:


# Create new event list and add it to observation object
new_event_list = EventList(table)
new_obs = observation.copy(in_memory=True, events=new_event_list)


# In[27]:


new_obs.events.table


# ## 4. Save new Event List and writing a modify HDU index table

# In the following, we show how to write the files in a directory contained in the original datastore directory. This follows the logic of DL3 data store and facilitate the manipulation of the HDU table.
# 
# If one does not want to save the events files and directly perform the pulsar analysis, this step is not required as well as the step of the meta data handling. However, be aware that for large dataset, the computation of phases can take tens of minutes. 

# In[28]:


data_store.hdu_table.base_dir


# In[29]:


# Define output directory and filename
datastore_dir = str(data_store.hdu_table.base_dir) + "/"
output_directory = "pulsar_events_file/"
output_path = datastore_dir + output_directory
filename = f"dl3_pulsar_{observation.obs_id:04d}.fits.gz"
file_path = output_path + filename

Path(output_path).mkdir(parents=True, exist_ok=True)


# In[30]:


output_path


# In[31]:


# Save the observation object in the specified file_path
print("Writing outputfile in " + str(file_path))
observation.events.write(
    filename=file_path, gti=observation.gti, overwrite=True
)


# Once the file has been written, we want to write a modified version of the HDU table. This is mandatory if we want to open the phased events file together with its associated IRFs. 

# In[32]:


# Print the current data store HDU table.
new_hdu = data_store.hdu_table.copy()
new_hdu


# In[33]:


for entry in new_hdu:
    if entry["HDU_NAME"] == "EVENTS" and entry["OBS_ID"] == observation.obs_id:
        entry["FILE_DIR"] = "./" + str(output_directory)
        entry["FILE_NAME"] = filename


# In[34]:


new_hdu


# We see that the `FILE_DIR`and `FILE_NAME`entry have been modified for our phased events file.

# Finally, we need to save the new HDU table in the origianl DL3 directory. Here one should be very careful to not name the new HDU file with the same name as the original HDU file of the data store. Otherwise, the original HDU file will be overwrited. 

# In[35]:


new_hdu.write(
    datastore_dir + "hdu-index-pulsar.fits.gz", format="fits", overwrite=True
)


# **Note: Here we use only one approach that could be useful, showing the steps to save the new Event files in a random directory and generate a new modified HDU index table. However, the user is free to chose the absolute path of the EventList and DataStore.  For instance, another approach could be making a full copy of the DataStore, or changing the location of the pulsar event files to one that could be more convinient for the user.**

# ## 5. Opening the new DataStore

# Once all of this is done, we just have to open the data store using `DataStore.from_dir()`and passing the pulsar HDU table to it :

# In[36]:


pulsar_datastore = DataStore.from_dir(
    DL3_direc, hdu_table_filename="hdu-index-pulsar.fits.gz"
)


# In[37]:


observations = pulsar_datastore.get_observations(
    [5029747], required_irf="point-like"
)
observations[0].available_hdus


# In[38]:


observations[0].events.table


# We can see that we recover both the IRFs and the events file with the phase column.

# ## 6. Pulsar analysis tools with gammapy

# Once we have the corret DataStore and the modified EventList with the phase information, we can do the pulsar analysis using different tools for Gammapy to compute the phaseogram, maps, SED, lightcurve, etc... To do so, one can check the following [Gammapy tutorial](https://docs.gammapy.org/1.0/tutorials/analysis-time/pulsar_analysis.html#sphx-glr-tutorials-analysis-time-pulsar-analysis-py).

# 
# Recipe made by [Alvaros Mas](https://github.com/alvmas), [Maxime Regeard](https://github.com/MRegeard), [Jan Lukas Schubert](https://github.com/jalu98).
