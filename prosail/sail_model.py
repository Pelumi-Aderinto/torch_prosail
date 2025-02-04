#!/usr/bin/env python
import numpy as np
import torch

from prosail import spectral_lib

from .FourSAIL import foursail
from .prospect_d import run_prospect, init_prospect_spectra, subsample_spectra

def init_prosail_spectra(
    soil_spectrum1:torch.Tensor|None=None,
    soil_spectrum2:torch.Tensor|None=None,
    nr:torch.Tensor|None=None,
    kab:torch.Tensor|None=None,
    kcar:torch.Tensor|None=None,
    kbrown:torch.Tensor|None=None,
    kw:torch.Tensor|None=None,
    km:torch.Tensor|None=None,
    kant:torch.Tensor|None=None,
    kprot:torch.Tensor|None=None,
    kcbc:torch.Tensor|None=None,
    lambdas:torch.Tensor|None=None,
    R_down:int=1,
    device:str='cpu', 
    method:str='block_mean', 
    prospect_version:str="5"):
    [nr, kab, kcar, kbrown, kw, km, kant, kprot, kcbc, 
     lambdas] = init_prospect_spectra(nr=nr,
                                    kab=kab,
                                    kcar=kcar,
                                    kbrown=kbrown,
                                    kw=kw,
                                    km=km,
                                    kant=kant,
                                    kprot=kprot,
                                    kcbc=kcbc,
                                    lambdas=lambdas,
                                    R_down=R_down, 
                                    prospect_version=prospect_version,
                                    device=device)
    soil_spectrum1 = spectral_lib.soil.rsoil1.to(device) if soil_spectrum1 is None else soil_spectrum1
    soil_spectrum2 = spectral_lib.soil.rsoil2.to(device) if soil_spectrum2 is None else soil_spectrum2
    if R_down > 1 :
        assert 2100 % R_down == 0
        soil_spectrum1 = subsample_spectra(soil_spectrum1, R_down=R_down, method=method)
        soil_spectrum2 = subsample_spectra(soil_spectrum2, R_down=R_down, method=method)
    return soil_spectrum1, soil_spectrum2, nr, kab, kcar, kbrown, kw, km, kant, kprot, kcbc,lambdas

def run_prosail(
    N:torch.Tensor, 
    cab:torch.Tensor, 
    car:torch.Tensor, 
    cbrown:torch.Tensor, 
    cw:torch.Tensor, 
    cm:torch.Tensor, 
    lai:torch.Tensor, 
    lidfa:torch.Tensor, 
    hspot:torch.Tensor, 
    rsoil:torch.Tensor, 
    psoil:torch.Tensor, 
    tts:torch.Tensor, 
    tto:torch.Tensor, 
    psi:torch.Tensor,
    ant:torch.Tensor|None=None, 
    prot:torch.Tensor|None=None, 
    cbc:torch.Tensor|None=None, 
    alpha=torch.as_tensor(40.0),
    prospect_version="5",
    typelidf=torch.as_tensor(2),
    lidfb=torch.as_tensor(0.0),
    factor="SDR",
    rsoil0=None,
    soil_spectrum1=None,
    soil_spectrum2=None,
    nr=None,
    kab=None,
    kcar=None,
    kbrown=None,
    kw=None,
    km=None,
    kant=None, 
    kprot=None, 
    kcbc=None,
    lambdas=None,
    device='cpu',
    R_down=1,
    init_spectra=False
):
    """Run the PROSPECT_5B and SAILh radiative transfer models. The soil
    model is a linear mixture model, where two spectra are combined together as

         rho_soil = rsoil*(psoil*soil_spectrum1+(1-psoil)*soil_spectrum2)
    By default, ``soil_spectrum1`` is a dry soil, and ``soil_spectrum2`` is a
    wet soil, so in that case, ``psoil`` is a surface soil moisture parameter.
    ``rsoil`` is a  soil brightness term. You can provide one or the two
    soil spectra if you want.  The soil spectra must be defined
    between 400 and 2500 nm with 1nm spacing.

    Parameters
    -----------
    1D tensors with prosail parameters as columns and batch as rows.
    parameters are :
    N: float
        The number of leaf layers. Unitless [-].
    cab: float
        The chlorophyll a+b concentration. [g cm^{-2}].
    car: float
        Carotenoid concentration.  [g cm^{-2}].
    cbrown: float
        The brown/senescent pigment. Unitless [-], often between 0 and 1
        but the literature on it is wide ranging!
    cw: float
        Equivalent leaf water. [cm]
    cm: float
        Dry matter [g cm^{-2}]
    lai: float
        leaf area index
    lidfa: float
        a parameter for leaf angle distribution. If ``typliedf``=2, average
        leaf inclination angle.
    tts: float
        Solar zenith angle
    tto: float
        Sensor zenith angle
    psi: float
        Relative sensor-solar azimuth angle ( saa - vaa )
    ant: float, optional
        Anthocyanins content. Used in Prospect-D and Prospect-PRO [g cm^{-2}]
    prot: float, optional
        Protein content. Used in Prospect-PRO. [g cm^{-2}]
    cbc: float, optional
        Carbon based constituents. Used in Prospect-PRO. [g cm^{-2}]
    alpha: float
        The alpha angle (in degrees) used in the surface scattering
        calculations. By default it's set to 40 degrees.
    prospect_version: str
        Which PROSPECT version to use. We have "5", "D" and "PRO"
    typelidf: int, optional
        The type of leaf angle distribution function to use. By default, is set
        to 2.
    lidfb: float, optional
        b parameter for leaf angle distribution. If ``typelidf``=2, ignored
    factor: str, optional
        What reflectance factor to return:
        * "SDR": directional reflectance factor (default)
        * "BHR": bi-hemispherical r. f.
        * "DHR": Directional-Hemispherical r. f. (directional illumination)
        * "HDR": Hemispherical-Directional r. f. (directional view)
        * "ALL": All of them
        * "ALLALL": All of the terms calculated by SAIL, including the above
    rsoil0: float, optional
        The soil reflectance spectrum
    rsoil: float, optional
        Soil scalar 1 (brightness)
    psoil: float, optional
        Soil scalar 2 (moisture)
    soil_spectrum1: 2101-element array
        First component of the soil spectrum
    soil_spectrum2: 2101-element array
        Second component of the soil spectrum
    Returns
    --------
    A reflectance factor between 400 and 2500 nm
    """    
    factor = factor.upper()
    if ant is None:
        ant = torch.zeros_like(cm).to(device)
    if prot is None:
        prot = torch.zeros_like(cm).to(device)
    if cbc is None:
        cbc = torch.zeros_like(cm).to(device)   
    if factor not in ["SDR", "BHR", "DHR", "HDR", "ALL", "ALLALL"]:
        raise ValueError(
            "'factor' must be one of SDR, BHR, DHR, HDR, ALL or ALLALL")
    if init_spectra:
        [soil_spectrum1, soil_spectrum2, nr, kab, kcar, kbrown, kw,
        km, kant, kprot, kcbc, lambdas] = init_prosail_spectra(soil_spectrum1=soil_spectrum1, 
                                                                soil_spectrum2=soil_spectrum2,
                                                                nr=nr,
                                                                kab=kab,
                                                                kcar=kcar,
                                                                kbrown=kbrown,
                                                                kw=kw,
                                                                km=km,
                                                                kant=kant, 
                                                                kprot=kprot, 
                                                                kcbc=kcbc,
                                                                lambdas=lambdas,
                                                                R_down=R_down, 
                                                                prospect_version=prospect_version,
                                                                device=device)
    elif nr is None:
        raise ValueError("Spectra must be initialized or be provided as input!")
    if rsoil0 is None:
        if (rsoil is None) or (psoil is None):
            raise ValueError("If rsoil0 isn't defined, then rsoil and psoil"
                             " need to be defined!")
        rsoil0 = rsoil * (psoil * soil_spectrum1 + (1.0 - psoil) * soil_spectrum2)
    else:
        rsoil0 = subsample_spectra(rsoil0, R_down=1)
    
    wv, refl, trans = run_prospect(
        N=N, 
        cab=cab, 
        car=car, 
        cbrown=cbrown, 
        cw=cw, 
        cm=cm,
        ant=ant,
        prot=prot,
        cbc=cbc,
        prospect_version=prospect_version,
        alpha=alpha,
        device=device,
        nr=nr,
        kab=kab,
        kcar=kcar,
        kbrown=kbrown,
        kw=kw,
        km=km,
        kant=kant,
        kprot=kprot,
        kcbc=kcbc,
        lambdas=lambdas,
        R_down=R_down, 
        init_spectra=False
    )

    [
        tss,
        too,
        tsstoo,
        rdd,
        tdd,
        rsd,
        tsd,
        rdo,
        tdo,
        rso,
        rsos,
        rsod,
        rddt,
        rsdt,
        rdot,
        rsodt,
        rsost,
        rsot,
        gammasdf,
        gammasdb,
        gammaso,
    ] = foursail(refl, trans, lidfa, lidfb, typelidf, lai, hspot, tts, tto,
                 psi, rsoil0)

    if factor == "SDR":
        return rsot
    elif factor == "BHR":
        return rddt
    elif factor == "DHR":
        return rsdt
    elif factor == "HDR":
        return rdot
    elif factor == "ALL":
        return [rsot, rddt, rsdt, rdot]
    elif factor == "ALLALL":
        return [
            tss,
            too,
            tsstoo,
            rdd,
            tdd,
            rsd,
            tsd,
            rdo,
            tdo,
            rso,
            rsos,
            rsod,
            rddt,
            rsdt,
            rdot,
            rsodt,
            rsost,
            rsot,
            gammasdf,
            gammasdb,
            gammaso,
        ]


def run_sail(
    refl,
    trans,
    lai,
    lidfa,
    hspot,
    tts,
    tto,
    psi,
    typelidf=2,
    lidfb=0.0,
    factor="SDR",
    rsoil0=None,
    rsoil=None,
    psoil=None,
    soil_spectrum1=None,
    soil_spectrum2=None,
):
    """Run the SAILh radiative transfer model. The soil model is a linear
    mixture model, where two spectra are combined together as

         rho_soil = rsoil*(psoil*soil_spectrum1+(1-psoil)*soil_spectrum2)

    By default, ``soil_spectrum1`` is a dry soil, and ``soil_spectrum2`` is a
    wet soil, so in that case, ``psoil`` is a surface soil moisture parameter.
    ``rsoil`` is a  soil brightness term. You can provide one or the two
    soil spectra if you want. The soil spectra, and leaf spectra must be defined
    between 400 and 2500 nm with 1nm spacing.

    Parameters
    ----------
    refl: 2101-element array
        Leaf reflectance
    trans: 2101-element array
        leaf transmittance
    lai: float
        leaf area index
    lidfa: float
        a parameter for leaf angle distribution. If ``typliedf``=2, average
        leaf inclination angle.
    hspot: float
        The hotspot parameter
    tts: float
        Solar zenith angle
    tto: float
        Sensor zenith angle
    psi: float
        Relative sensor-solar azimuth angle ( saa - vaa )
    typelidf: int, optional
        The type of leaf angle distribution function to use. By default, is set
        to 2.
    lidfb: float, optional
        b parameter for leaf angle distribution. If ``typelidf``=2, ignored
    factor: str, optional
        What reflectance factor to return:
        * "SDR": directional reflectance factor (default)
        * "BHR": bi-hemispherical r. f.
        * "DHR": Directional-Hemispherical r. f. (directional illumination)
        * "HDR": Hemispherical-Directional r. f. (directional view)
        * "ALL": All of them
        * "ALLALL": All of the terms calculated by SAIL, including the above
    rsoil0: float, optional
        The soil reflectance spectrum
    rsoil: float, optional
        Soil scalar 1 (brightness)
    psoil: float, optional
        Soil scalar 2 (moisture)
    soil_spectrum1: 2101-element array
        First component of the soil spectrum
    soil_spectrum2: 2101-element array
        Second component of the soil spectrum

    Returns
    --------
    Directional surface reflectance between 400 and 2500 nm


    """

    factor = factor.upper()
    if factor not in ["SDR", "BHR", "DHR", "HDR", "ALL", "ALLALL"]:
        raise ValueError(
            "'factor' must be one of SDR, BHR, DHR, HDR, ALL or ALLALL")
    if soil_spectrum1 is not None:
        assert len(soil_spectrum1) == 2101
    else:
        soil_spectrum1 = spectral_lib.soil.rsoil1

    if soil_spectrum2 is not None:
        assert len(soil_spectrum1) == 2101
    else:
        soil_spectrum2 = spectral_lib.soil.rsoil2

    if rsoil0 is None:
        if (rsoil is None) or (psoil is None):
            raise ValueError("If rsoil0 isn't define, then rsoil and psoil"
                             " need to be defined!")
        rsoil0 = rsoil * (psoil * soil_spectrum1 +
                          (1.0 - psoil) * soil_spectrum2)

    [
        tss,
        too,
        tsstoo,
        rdd,
        tdd,
        rsd,
        tsd,
        rdo,
        tdo,
        rso,
        rsos,
        rsod,
        rddt,
        rsdt,
        rdot,
        rsodt,
        rsost,
        rsot,
        gammasdf,
        gammasdb,
        gammaso,
    ] = foursail(refl, trans, lidfa, lidfb, typelidf, lai, hspot, tts, tto,
                 psi, rsoil0)

    if factor == "SDR":
        return rsot
    elif factor == "BHR":
        return rddt
    elif factor == "DHR":
        return rsdt
    elif factor == "HDR":
        return rdot
    elif factor == "ALL":
        return [rsot, rddt, rsdt, rdot]
    elif factor == "ALLALL":
        return [
            tss,
            too,
            tsstoo,
            rdd,
            tdd,
            rsd,
            tsd,
            rdo,
            tdo,
            rso,
            rsos,
            rsod,
            rddt,
            rsdt,
            rdot,
            rsodt,
            rsost,
            rsot,
            gammasdf,
            gammasdb,
            gammaso,
        ]


def run_thermal_sail(
    lam,
    tveg,
    tsoil,
    tveg_sunlit,
    tsoil_sunlit,
    t_atm,
    lai,
    lidfa,
    hspot,
    tts,
    tto,
    psi,
    rsoil=None,
    refl=None,
    emv=None,
    ems=None,
    typelidf=2,
    lidfb=0,
):
    c1 = 3.741856e-16
    c2 = 14388.0
    # Calculate the thermal emission from the different
    # components using Planck's Law
    top = (1.0e-6) * c1 * (lam * 1e-6)**(-5.0)
    Hc = top / (torch.exp(c2 / (lam * tveg)) - 1.0)  # Shade leaves
    Hh = top / (torch.exp(c2 / (lam * tveg_sunlit)) - 1.0)  # Sunlit leaves
    Hd = top / (torch.exp(c2 / (lam * tsoil)) - 1.0)  # shade soil
    Hs = top / (torch.exp(c2 / (lam * tsoil_sunlit)) - 1.0)  # Sunlit soil
    Hsky = top / (torch.exp(c2 / (lam * t_atm)) - 1.0)  # Sky emission

    # Emissivity calculations
    if refl is not None and emv is None:
        emv = 1.0 - refl  # Assuming absorption is 1

    if rsoil is not None and ems is None:
        ems = 1.0 - rsoil

    if rsoil is None and ems is not None:
        rsoil = 1.0 - ems
    if refl is None and emv is not None:
        refl = 1.0 - emv

    [
        tss,
        too,
        tsstoo,
        rdd,
        tdd,
        rsd,
        tsd,
        rdo,
        tdo,
        rso,
        rsos,
        rsod,
        rddt,
        rsdt,
        rdot,
        rsodt,
        rsost,
        rsot,
        gammasdf,
        gammasdb,
        gammaso,
    ] = foursail(
        refl,
        torch.zeros_like(refl),
        lidfa,
        lidfb,
        typelidf,
        lai,
        hspot,
        tts,
        tto,
        psi,
        rsoil,
    )

    gammad = 1.0 - rdd - tdd
    gammao = 1.0 - rdo - tdo - too

    tso = tss * too + tss * (tdo + rsoil * rdd * too) / (1.0 - rsoil * rdd)
    ttot = (too + tdo) / (1.0 - rsoil * rdd)
    gammaot = gammao + ttot * rsoil * gammad
    gammasot = gammaso + ttot * rsoil * gammasdf

    aeev = gammaot
    aees = ttot * ems

    Lw = (rdot * Hsky +
          (aeev * Hc + gammasot * emv * (Hh - Hc) + aees * Hd + tso * ems *
           (Hs - Hd))) / np.pi

    dnoem1 = top / (Lw * np.pi)
    Tbright = c2 / (lam * torch.log(dnoem1 + 1.0))
    dir_em = 1.0 - rdot
    return Lw, Tbright, dir_em
