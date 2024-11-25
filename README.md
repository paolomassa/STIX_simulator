This repository contains a python code for simulating X-ray flaring configurations and corresponding data recorded by the [Spectrometer/Telescope for Imaging X-rays (STIX)](https://datacenter.stix.i4ds.net/) on board the ESA Solar Orbiter mission.
The simulation code is based on **"The STIX Imaging Concept"** (2023) by Massa et al. [(https://doi.org/10.1007/s11207-023-02205-7)](https://doi.org/10.1007/s11207-023-02205-7).

Specifically, `simulator.py` contains the following functions.

- `SimulateConfig` for simulating a random X-ray flaring configuration consisting of a number of sources which is provided as input by the user. 
This function returns both the parameter values describing the configuration and the corresponding STIX data.
- `CreateGaussianSource` for generating the image of a Gaussian-shaped X-ray source from the corresponding parameters.

An example of the simulation code is provided in `demo_simulator.ipynb`.
