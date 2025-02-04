#!/usr/bin/env python
"""Spectral libraries for PROSPECT + SAIL
"""
import pkgutil
from collections import namedtuple
from io import BytesIO

import torch
import numpy as np
# import warnings

Spectra = namedtuple("Spectra", "prospect5 prospectd prospectpro soil light")
Prospect5Spectra = namedtuple("Prospect5Spectra", "nr kab kcar kbrown kw km")
ProspectDSpectra = namedtuple("ProspectDSpectra",
                              "nr kab kcar kbrown kw km kant")
ProspectPROSpectra = namedtuple("ProspectPROSpectra",
                                "nr kab kcar kbrown kw km kant kprot kcbc")
SoilSpectra = namedtuple("SoilSpectra", "rsoil1 rsoil2")
LightSpectra = namedtuple("LightSpectra", "es ed")


def get_spectra():
    """Reads the spectral information and stores is for future use."""
    # warnings.warn("WARNING: Spectra data should only be read at intialization.")
    # PROSPECT-D
    prospect_d_spectraf = pkgutil.get_data("prosail", "prospect_d_spectra.txt")
    _, nr, kab, kcar, kant, kbrown, kw, km = torch.from_numpy(
        np.loadtxt(BytesIO(prospect_d_spectraf), unpack=True)).float()
    prospect_d_spectra = ProspectDSpectra(nr, kab, kcar, kbrown, kw, km, kant)
    # PROSPECT-PRO
    prospect_pro_spectraf = pkgutil.get_data("prosail",
                                             "prospect_pro_spectra.txt")
    _, nr, kab, kcar, kant, kbrown, kw, km, kprot, kcbc = torch.from_numpy(
        np.loadtxt(BytesIO(prospect_pro_spectraf), unpack=True)).float()
    prospect_pro_spectra = ProspectPROSpectra(nr, kab, kcar, kbrown, kw, km,
                                              kant, kprot, kcbc)
    # PROSPECT 5
    prospect_5_spectraf = pkgutil.get_data("prosail", "prospect5_spectra.txt")
    nr, kab, kcar, kbrown, kw, km = torch.from_numpy(
        np.loadtxt(BytesIO(prospect_5_spectraf), unpack=True)).float()
    prospect_5_spectra = Prospect5Spectra(nr, kab, kcar, kbrown, kw, km)
    # SOIL
    soil_spectraf = pkgutil.get_data("prosail", "soil_reflectance.txt")
    rsoil1, rsoil2 = torch.from_numpy(
        np.loadtxt(BytesIO(soil_spectraf), unpack=True)).float()
    soil_spectra = SoilSpectra(rsoil1, rsoil2)
    # LIGHT
    light_spectraf = pkgutil.get_data("prosail", "light_spectra.txt")
    es, ed = torch.from_numpy(np.loadtxt(BytesIO(light_spectraf), unpack=True)).float()
    light_spectra = LightSpectra(es, ed)
    spectra = Spectra(
        prospect_5_spectra,
        prospect_d_spectra,
        prospect_pro_spectra,
        soil_spectra,
        light_spectra,
    )
    return spectra
