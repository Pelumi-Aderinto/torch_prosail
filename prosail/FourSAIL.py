#!/usr/bin/env python

from typing import Tuple

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

import numpy as np
import torch



# @numba.jit("Tuple((f8, f8, f8, f8))(f8,f8,f8,f8)", nopython=True, cache=True)
@torch.jit.script
def volscatt(
    tts: torch.Tensor, tto: torch.Tensor, psi: torch.Tensor, ttl: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute volume scattering functions and interception coefficients
    for given solar zenith, viewing zenith, azimuth and leaf inclination angle.

    Parameters
    ----------
    tts : float
        Solar Zenith Angle (degrees).
    tto : float
        View Zenight Angle (degrees).
    psi : float
        View-Sun reliative azimuth angle (degrees).
    ttl : float
        leaf inclination angle (degrees).

    Returns
    -------
    chi_s : float
        Interception function  in the solar path.
    chi_o : float
        Interception function  in the view path.
    frho : float
        Function to be multiplied by leaf reflectance to obtain the volume scattering.
    ftau : float
        Function to be multiplied by leaf transmittance to obtain the volume scattering.

    References
    ----------
    Wout Verhoef, april 2001, for CROMA.
    """

    cts = torch.cos(torch.deg2rad(tts))
    cto = torch.cos(torch.deg2rad(tto))
    sts = torch.sin(torch.deg2rad(tts))
    sto = torch.sin(torch.deg2rad(tto))
    cospsi = torch.cos(torch.deg2rad(psi))
    psir = torch.deg2rad(psi)
    cttl = torch.cos(torch.deg2rad(ttl))
    sttl = torch.sin(torch.deg2rad(ttl))
    cs = cttl * cts
    co = cttl * cto
    ss = sttl * sts
    so = sttl * sto
    cosbts = 5.0 * torch.ones_like(ss)
    ss_sup_1em6 = ss > 1e-6
    cosbts[ss_sup_1em6] = -cs[ss_sup_1em6] / ss[ss_sup_1em6]

    cosbto = 5.0 * torch.ones_like(so)
    so_sup_1em6 = so > 1e-6
    cosbto[so_sup_1em6] = -co[so_sup_1em6] / so[so_sup_1em6]
    
    bts = torch.as_tensor(np.pi).to(so.device) * torch.ones_like(so)
    ds = 1.0 * cs
    cosbts_inf_1 = torch.abs(cosbts) < 1.0
    bts[cosbts_inf_1] = torch.arccos(cosbts[cosbts_inf_1])
    ds[cosbts_inf_1] = ss[cosbts_inf_1]
    
    chi_s = 2.0 / np.pi * ((bts - np.pi * 0.5) * cs + torch.sin(bts) * ss)
    bto = torch.zeros_like(cosbto)
    do_ = -co
    
    tto_inf_90 = (tto < 90.0).repeat(1,co.size(1))
    bto[tto_inf_90] = torch.as_tensor(np.pi).to(so.device)
    do_[tto_inf_90] = co[tto_inf_90]
    
    cosbto_inf_1 = torch.abs(cosbto) < 1.0
    bto[cosbto_inf_1] = torch.arccos(cosbto[cosbto_inf_1])
    do_[cosbto_inf_1] = so[cosbto_inf_1]
    
    chi_o = 2.0 / np.pi * ((bto - np.pi * 0.5) * co + torch.sin(bto) * so)
    btran1 = torch.abs(bts - bto)
    btran2 = np.pi - torch.abs(bts + bto - np.pi)
    
    bt1 = 1.0 * btran1
    bt2 = 1.0 * btran2
    bt3 = psir.repeat(1,btran2.size(1))
    
    psir_infeql_btran1 = psir <= btran1
    psir_infeql_btran2 = psir <= btran2
    
    bt2[psir_infeql_btran2] = psir.repeat(1,btran2.size(1))[psir_infeql_btran2]
    bt3[psir_infeql_btran2] = btran2[psir_infeql_btran2]
    
    bt1[psir_infeql_btran1] = psir.repeat(1,btran2.size(1))[psir_infeql_btran1]
    bt2[psir_infeql_btran1] = btran1[psir_infeql_btran1]
    bt3[psir_infeql_btran1] = btran2[psir_infeql_btran1]
    
    t1 = 2.0 * cs * co + ss * so * cospsi
    t2 = torch.zeros_like(bt2)
    bt2_sup_0 = bt2 > 0.0
    t2[bt2_sup_0] = torch.sin(bt2[bt2_sup_0]) * (2.0 * ds[bt2_sup_0] * do_[bt2_sup_0] +
                           ss[bt2_sup_0] * so[bt2_sup_0] * torch.cos(bt1[bt2_sup_0]) * torch.cos(bt3[bt2_sup_0]))

    denom = 2.0 * np.pi**2
    frho = torch.max(((np.pi - bt2) * t1 + t2) / denom, torch.zeros_like(bt2))
    ftau = torch.max((-bt2 * t1 + t2) / denom, torch.zeros_like(bt2))

    return (chi_s, chi_o, frho, ftau)


# @numba.jit(
#     "Tuple((f8, f8, f8, f8, f8))(f8[:], f8, f8, f8)",
#     nopython=True,
#     cache=True,
# )
@torch.jit.script
def weighted_sum_over_lidf(
    lidf: torch.Tensor, tts: torch.Tensor, tto: torch.Tensor, psi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    ks = torch.zeros_like(lidf)
    ko = torch.zeros_like(lidf)
    bf = torch.zeros_like(lidf)
    sob = torch.zeros_like(lidf)
    sof = torch.zeros_like(lidf)
    cts = torch.cos(torch.deg2rad(tts))
    cto = torch.cos(torch.deg2rad(tto))
    ctscto = cts * cto

    n_angles = lidf.size(1)
    angle_step = float(90.0 / n_angles)
    litab = (torch.arange(n_angles) * angle_step + (angle_step * 0.5)).repeat(lidf.size(0),1).to(tts.device)
    cttl = torch.cos(torch.deg2rad(litab))
    chi_s, chi_o, frho, ftau = volscatt(tts, tto, psi, litab)
    ksli = chi_s / cts
    koli = chi_o / cto
    # Area scattering coefficient fractions
    sobli = frho * np.pi / ctscto
    sofli = ftau * np.pi / ctscto
    bfli = cttl**2.0
    ks = (ksli * lidf).sum(1).reshape(-1,1)
    ko = (koli * lidf).sum(1).reshape(-1,1)
    bf = (bfli * lidf).sum(1).reshape(-1,1)
    sob = (sobli * lidf).sum(1).reshape(-1,1)
    sof = (sofli * lidf).sum(1).reshape(-1,1)
    return ks, ko, bf, sob, sof


@lru_cache(maxsize=16)
def define_geometric_constants(tts, tto, psi):
    cts = torch.cos(torch.deg2rad(tts))
    cto = torch.cos(torch.deg2rad(tto))
    ctscto = cts * cto
    tants = torch.tan(torch.deg2rad(tts))
    tanto = torch.tan(torch.deg2rad(tto))
    cospsi = torch.cos(torch.deg2rad(psi))
    dso = torch.sqrt(tants**2.0 + tanto**2.0 - 2.0 * tants * tanto * cospsi)
    return cts, cto, ctscto, tants, tanto, cospsi, dso


#@numba.jit("Tuple((f8,f8))(f8,f8,f8,f8)", nopython=True, cache=True)
@torch.jit.script
def hotspot_calculations(
        alf: torch.Tensor, lai: torch.Tensor, ko: torch.Tensor,
        ks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    fhot = lai * torch.sqrt(ko * ks)
    # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
    n_steps = 20
    x2 = torch.ones(alf.size(0), n_steps).to(alf.device)
    fint = ((1.0 - torch.exp(-alf)) * 0.05).repeat(1,n_steps)
    sumint = torch.zeros(alf.size(0), n_steps).to(alf.device)
    x2[:,:-1] = -torch.log(1.0 - torch.arange(1,n_steps).unsqueeze(0).to(alf.device) * fint[:,:-1]) / alf
    y2 = (-(ko + ks) * lai).reshape(-1,1) * x2 + fhot.reshape(-1,1) * (1.0 - torch.exp(-alf * x2)) / alf
    f2 = torch.exp(y2)
    
    x1 = torch.zeros(alf.size(0), n_steps).to(alf.device)
    x1[:,1:] = x2[:,:-1]
    y1 = torch.zeros(alf.size(0), n_steps).to(alf.device)
    y1[:,1:] = y2[:,:-1]
    f1 = torch.ones(alf.size(0), n_steps).to(alf.device)
    f1[:,1:] = f2[:,:-1]
    EPS=torch.tensor(1e-8)
    denom = torch.sign(y2 - y1) * ((y2 - y1).abs() + EPS)
    sumint = ((f2 - f1) * (x2 - x1) / denom).sum(1)
    tsstoo = f2[:,-1]
    return tsstoo, sumint

# @torch.jit.script
def Jfunc1(k, l, t):
    """ J1 function with avoidance of singularity problem."""
    try:
        nb = l.size(1)
    except TypeError:
        nb = 1
    k_ext = k.repeat(1, nb)
    t_ext = t.repeat(1, nb)
    del_ = (k - l) * t
    if nb > 1:
        del_sup_1em3 = torch.abs(del_) > 1e-3
        result = (
            0.5 * t_ext *
            (torch.exp(-k_ext * t_ext) + torch.exp(-l * t_ext)) *
            (1.0 - (del_**2.0) / 12.0))
        result[del_sup_1em3] = (
            torch.exp((-l * t_ext)[del_sup_1em3]) - torch.exp(
                -k_ext * t_ext)[del_sup_1em3]) / ((k_ext - l)[del_sup_1em3])
    return result

@torch.jit.script
def Jfunc2(k, l, t):
    """J2 function."""
    return (1.0 - torch.exp(-(k + l) * t)) / (k + l)


#@numba.jit("f8[:](f8,f8,i8)", nopython=True, cache=True)
@torch.jit.script
def verhoef_bimodal(a: torch.Tensor,
                    b: torch.Tensor,
                    n_elements: int = 18) -> torch.Tensor:
    """Calculate the Leaf Inclination Distribution Function based on the
    Verhoef's bimodal LIDF distribution.
    Parameters
    ----------
    a : float
        controls the average leaf slope.
    b : float
        controls the distribution's bimodality.

            * LIDF type     [a,b].
            * Planophile    [1,0].
            * Erectophile   [-1,0].
            * Plagiophile   [0,-1].
            * Extremophile  [0,1].
            * Spherical     [-0.35,-0.15].
            * Uniform       [0,0].
            * requirement: |LIDFa| + |LIDFb| < 1.
    n_elements : int
        Total number of equally spaced inclination angles.

    Returns
    -------
    lidf : list
        Leaf Inclination Distribution Function at equally spaced angles.

    References
    ----------
    .. [Verhoef1998] Verhoef, Wout. Theory of radiative transfer models applied
        in optical remote sensing of vegetation canopies.
        Nationaal Lucht en Ruimtevaartlaboratorium, 1998.
        http://library.wur.nl/WebQuery/clc/945481.
    """
    freq = torch.as_tensor(1.0)
    step = 90.0 / n_elements
    lidf = torch.zeros(n_elements, dtype=torch.double) * 1.0
    angles = torch.flip(torch.arange(n_elements) * step, [0])
    i = 0
    for angle in angles:
        tl1 = torch.deg2rad(torch.as_tensor(angle, dtype=torch.double))
        if a > 1.0:
            f = 1.0 - torch.cos(tl1)
        else:
            eps = 1e-8
            delx = torch.as_tensor(1.0)
            x = 2.0 * tl1
            p = x
            y = torch.sin(x)
            while delx >= eps:
                y = a * torch.sin(x) + 0.5 * b * torch.sin(2.0 * x)
                dx = 0.5 * (y - x + p)
                x = x + dx
                delx = abs(dx)
            f = (2.0 * y + p) / np.pi
        freq = freq - f
        lidf[i] = freq
        freq = f
        i += 1
    lidf = torch.flip(lidf, [0])
    return lidf


# @numba.jit("f8[:](f8,i8)", nopython=True, cache=True)
@torch.jit.script
def campbell(alpha: torch.Tensor, n_elements: int = 18) -> torch.Tensor:
    """Calculate the Leaf Inclination Distribution Function based on the
    mean angle of [Campbell1990] ellipsoidal LIDF distribution.
    Parameters
    ----------
    alpha : float
        Mean leaf angle (degrees) use 57 for a spherical LIDF.
    n_elements : int
        Total number of equally spaced inclination angles .

    Returns
    -------
    lidf : list
        Leaf Inclination Distribution Function for 18 equally spaced angles.

    References
    ----------
    .. [Campbell1986] G.S. Campbell, Extinction coefficients for radiation in
        plant canopies calculated using an ellipsoidal inclination angle distribution,
        Agricultural and Forest Meteorology, Volume 36, Issue 4, 1986, Pages 317-321,
        ISSN 0168-1923, http://dx.doi.org/10.1016/0168-1923(86)90010-9.
    .. [Campbell1990] G.S Campbell, Derivation of an angle density function for
        canopies with ellipsoidal leaf angle distributions,
        Agricultural and Forest Meteorology, Volume 49, Issue 3, 1990, Pages 173-176,
        ISSN 0168-1923, http://dx.doi.org/10.1016/0168-1923(90)90030-A.
    """
    excent = torch.exp(-1.6184e-5 * alpha**3.0 + 2.1145e-3 * alpha**2.0 -
                       1.2390e-1 * alpha + 3.2491).reshape(-1,1)
    sum0 = torch.tensor(0.0)
    freq = torch.zeros(alpha.size(0), n_elements).to(alpha.device)
    step = 90.0 / n_elements
    degrees = torch.arange(0, n_elements+1).repeat(alpha.size(0),1).to(alpha.device) * step
    tl1 = torch.deg2rad(degrees[:,:-1])
    tl2 = torch.deg2rad(degrees[:,1:])
    x1 = excent / (torch.sqrt(1.0 + excent**2.0 * torch.tan(tl1)**2.0))
    x2 = excent / (torch.sqrt(1.0 + excent**2.0 * torch.tan(tl2)**2.0))
    EPS=torch.tensor(1e-8) # Quickfix for alph divergence with excent = 1 (alpha approx 58.435)
    alph = excent / (torch.sqrt(torch.abs(1.0 - excent**2.0)+ EPS))
    alph2 = alph**2.0
    x12 = x1**2.0
    x22 = x2**2.0
    alsx1 = torch.sqrt(alph2 + torch.sign(excent - 1) * x12)
    alsx2 = torch.sqrt(alph2 + torch.sign(excent - 1) * x22)
    dum =  x1 * alsx1

    excent_eql_1 = torch.where(excent==1)[0]
    if len(excent_eql_1)>0:
        freq[excent_eql_1,:] = torch.abs(torch.cos(tl1[excent_eql_1]) - torch.cos(tl2[excent_eql_1]))
    
    excent_sup_1 = torch.where(excent>1)[0]
    if len(excent_sup_1)>0:
        dum[excent_sup_1,:] = dum[excent_sup_1,:] + alph2[excent_sup_1] * torch.log(x1[excent_sup_1,:] + alsx1[excent_sup_1,:])
        freq[excent_sup_1,:] = torch.abs(dum[excent_sup_1,:] -
                      (x2[excent_sup_1,:] * alsx2[excent_sup_1,:] + alph2[excent_sup_1] * torch.log(x2[excent_sup_1,:] + alsx2[excent_sup_1,:])))
        
    excent_inf_1 = torch.where(excent<1)[0]
    if len(excent_inf_1)>0:
        dum[excent_inf_1,:] = dum[excent_inf_1,:] + alph2[excent_inf_1] * torch.arcsin(x1[excent_inf_1,:] / alph[excent_inf_1])
        freq[excent_inf_1,:] = torch.abs(dum[excent_inf_1,:] -
                      (x2[excent_inf_1,:] * alsx2[excent_inf_1,:] + 
                       alph2[excent_inf_1] * torch.arcsin(x2[excent_inf_1,:] / alph[excent_inf_1])))
    
    sum0 = freq.sum(1).unsqueeze(1).repeat(1, n_elements)

    lidf = freq / sum0

    return lidf


def foursail(rho, tau, lidfa, lidfb, lidftype, lai, hotspot, tts, tto, psi,
             rsoil):
    """
    Parameters
    ----------
    rho : array_like
        leaf lambertian reflectance.
    tau : array_like
        leaf transmittance.
    lidfa : float
        Leaf Inclination Distribution at regular angle steps.
    lidfb : float
        Leaf Inclination Distribution at regular angle steps.
    lidftype : float
        Leaf Inclination Distribution at regular angle steps.
    lai : float
        Leaf Area Index.
    hotspot : float
        Hotspot parameter.
    tts : float
        Sun Zenith Angle (degrees).
    tto : float
        View(sensor) Zenith Angle (degrees).
    psi : float
        Relative Sensor-Sun Azimuth Angle (degrees).
    rsoil : array_like
        soil lambertian reflectance.

    Returns
    -------
    tss : array_like
        beam transmittance in the sun-target path.
    too : array_like
        beam transmittance in the target-view path.
    tsstoo : array_like
        beam tranmittance in the sur-target-view path.
    rdd : array_like
        canopy bihemisperical reflectance factor.
    tdd : array_like
        canopy bihemishperical transmittance factor.
    rsd : array_like
        canopy directional-hemispherical reflectance factor.
    tsd : array_like
        canopy directional-hemispherical transmittance factor.
    rdo : array_like
        canopy hemispherical-directional reflectance factor.
    tdo : array_like
        canopy hemispherical-directional transmittance factor.
    rso : array_like
        canopy bidirectional reflectance factor.
    rsos : array_like
        single scattering contribution to rso.
    rsod : array_like
        multiple scattering contribution to rso.
    rddt : array_like
        surface bihemispherical reflectance factor.
    rsdt : array_like
        surface directional-hemispherical reflectance factor.
    rdot : array_like
        surface hemispherical-directional reflectance factor.
    rsodt : array_like
        reflectance factor.
    rsost : array_like
        reflectance factor.
    rsot : array_like
        surface bidirectional reflectance factor.
    gammasdf : array_like
        'Thermal gamma factor'.
    gammasdb : array_like
        'Thermal gamma factor'.
    gammaso : array_like
        'Thermal gamma factor'.

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844 based on  in Verhoef et al. (2007).
    """
    # Define some geometric constants.
    cts, cto, ctscto, tants, tanto, cospsi, dso = define_geometric_constants(
        tts, tto, psi)

    # Calcualte leaf angle distribution
    if lidftype == 1:
        raise NotImplementedError("Vectorizing this equation is impossible for the moment")
        lidf = verhoef_bimodal(lidfa, lidfb, n_elements=18)
    elif lidftype == 2:
        lidf = campbell(lidfa, n_elements=18)
    else:
        raise ValueError(
            "lidftype can only be 1 (Campbell) or 2 (ellipsoidal)")
    # Calculate geometric factors associated with extinction and scattering
    ks, ko, bf, sob, sof = weighted_sum_over_lidf(lidf, tts, tto, psi)

    # Geometric factors to be used later with rho and tau
    sdb = (0.5 * (ks + bf)).repeat(1,rho.size(1))
    sdf = (0.5 * (ks - bf)).repeat(1,rho.size(1))
    dob = (0.5 * (ko + bf)).repeat(1,rho.size(1))
    dof = (0.5 * (ko - bf)).repeat(1,rho.size(1))
    ddb = (0.5 * (1.0 + bf)).repeat(1,rho.size(1))
    ddf = (0.5 * (1.0 - bf)).repeat(1,rho.size(1))

    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    sigf = torch.max(torch.tensor(1e-36), sigf)
    sigb = torch.max(torch.tensor(1e-36), sigb)
    
    att = 1.0 - sigf
    m = torch.sqrt(att**2.0 - sigb**2.0)
    sb = sdb * rho + sdf * tau
    sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau
    w = sob * rho + sof * tau

    tss = torch.ones_like(sob).float()
    too = torch.ones_like(sob).float()
    tsstoo = torch.ones_like(sob).float()
    rdd = torch.zeros_like(m).float()
    tdd = torch.ones_like(m).float()
    rsd = torch.zeros_like(m).float()
    tsd = torch.zeros_like(m).float()
    rdo = torch.zeros_like(m).float()
    tdo = torch.zeros_like(m).float()
    rso = torch.zeros_like(m).float()
    rsos = torch.zeros_like(m).float()
    rsod = torch.zeros_like(m).float()
    rddt = rsoil.clone()
    rsdt = rsoil.clone()
    rdot = rsoil.clone()
    rsodt = torch.zeros_like(m).float()
    rsost = rsoil.clone()
    rsot = rsoil.clone()
    gammasdf = torch.zeros_like(m)
    gammaso = torch.zeros_like(m)
    gammasdb = torch.zeros_like(m)
    lai_sup_0 = torch.where(lai>0)[0]
    if len(lai_sup_0) > 0 :
        
        e1 = torch.exp(-m[lai_sup_0,:] * lai[lai_sup_0])
        e2 = e1**2.0
        rinf = (att[lai_sup_0,:] - m[lai_sup_0,:]) / sigb[lai_sup_0,:]
        rinf2 = rinf**2.0
        re = rinf * e1
        denom = 1.0 - rinf2 * e2
        J1ks = Jfunc1(ks[lai_sup_0], m[lai_sup_0,:], lai[lai_sup_0])
        J2ks = Jfunc2(ks[lai_sup_0], m[lai_sup_0,:], lai[lai_sup_0])
        J1ko = Jfunc1(ko[lai_sup_0], m[lai_sup_0,:], lai[lai_sup_0])
        J2ko = Jfunc2(ko[lai_sup_0], m[lai_sup_0,:], lai[lai_sup_0])
        Pss = (sf[lai_sup_0,:] + sb[lai_sup_0,:] * rinf) * J1ks
        Qss = (sf[lai_sup_0,:] * rinf + sb[lai_sup_0,:]) * J2ks
        Pv = (vf[lai_sup_0,:] + vb[lai_sup_0,:] * rinf) * J1ko
        Qv = (vf[lai_sup_0,:] * rinf + vb[lai_sup_0,:]) * J2ko
        tdd[lai_sup_0,:] = (1.0 - rinf2) * e1 / denom
        rdd[lai_sup_0,:] = rinf * (1.0 - e2) / denom
        tsd[lai_sup_0,:] = (Pss - re * Qss) / denom
        rsd[lai_sup_0,:] = (Qss - re * Pss) / denom
        tdo[lai_sup_0,:] = (Pv - re * Qv) / denom
        rdo[lai_sup_0,:] = (Qv - re * Pv) / denom
        # Thermal "sd" quantities
        gammasdf[lai_sup_0,:] = (1.0 + rinf) * (J1ks - re * J2ks) / denom
        gammasdb[lai_sup_0,:] = (1.0 + rinf) * (-re * J1ks + J2ks) / denom
        tss[lai_sup_0] = torch.exp(-ks[lai_sup_0] * lai[lai_sup_0])
        too[lai_sup_0] = torch.exp(-ko[lai_sup_0] * lai[lai_sup_0])
        z = Jfunc2(ks[lai_sup_0], ko[lai_sup_0], lai[lai_sup_0])
        g1 = (z - J1ks * too[lai_sup_0]) / (ko[lai_sup_0] + m[lai_sup_0,:])
        g2 = (z - J1ko * tss[lai_sup_0]) / (ks[lai_sup_0] + m[lai_sup_0,:])
        Tv1 = (vf[lai_sup_0,:] * rinf + vb[lai_sup_0,:]) * g1
        Tv2 = (vf[lai_sup_0,:] + vb[lai_sup_0,:] * rinf) * g2
        T1 = Tv1 * (sf[lai_sup_0,:] + sb[lai_sup_0,:] * rinf)
        T2 = Tv2 * (sf[lai_sup_0,:] * rinf + sb[lai_sup_0,:])
        T3 = (rdo[lai_sup_0,:] * Qss + tdo[lai_sup_0,:] * Pss) * rinf
        # Multiple scattering contribution to bidirectional canopy reflectance
        rsod = (T1 + T2 - T3) / (1.0 - rinf2)
        # Thermal "sod" quantity
        T4 = Tv1 * (1.0 + rinf)
        T5 = Tv2 * (1.0 + rinf)
        T6 = (rdo[lai_sup_0,:] * J2ks + tdo[lai_sup_0,:] * J1ks) * (1.0 + rinf) * rinf
        gammasod = (T4 + T5 - T6) / (1.0 - rinf2)
        # Treatment of the hotspot-effect
        alf = torch.as_tensor(1e36) * torch.ones_like(tss).float()
        # Apply correction 2/(K+k) suggested by F.-M. Breon
        hotspot_sup_0 = torch.where((hotspot > 0.0)*(lai>0))[0]
        if len(hotspot_sup_0)>0:
            alf[hotspot_sup_0] = (dso[hotspot_sup_0] / hotspot[hotspot_sup_0]) * 2.0 / (ks[hotspot_sup_0] + ko[hotspot_sup_0])
        
        alf_eql_0 = torch.where((alf==0.0)*(lai>0))[0]
        sumint = torch.zeros_like(tss).float()
        if len(alf_eql_0)>0:
            tsstoo[alf_eql_0] = tss[alf_eql_0]
            sumint[alf_eql_0] = (1.0 - tss[alf_eql_0]) / (ks[alf_eql_0] * lai[alf_eql_0])
        
        alf_nql_0 = torch.where((alf!=0.0)*(lai>0))[0]
        if len(alf!=0.0)>0:
            tsstoo_n0, sumint_n0 = hotspot_calculations(alf[alf_nql_0], lai[alf_nql_0],
                                                        ko[alf_nql_0], ks[alf_nql_0])
            tsstoo[alf_nql_0] = tsstoo_n0.reshape(-1,1)
            sumint[alf_nql_0] = sumint_n0.reshape(-1,1)
    
        # Bidirectional reflectance
        # Single scattering contribution
        rsos = w[lai_sup_0,:] * lai[lai_sup_0] * sumint[lai_sup_0]
        gammasos = ko[lai_sup_0] * lai[lai_sup_0] * sumint[lai_sup_0]
        # Total canopy contribution
        rso = rsos + rsod
        gammaso = gammasos + gammasod
        # Interaction with the soil
        dn = 1.0 - rsoil[lai_sup_0,:] * rdd[lai_sup_0,:]
        # try:
        #     dn[dn < 1e-36] = 1e-36
        # except TypeError:
        if (dn < 1e-8).any():
            pass
        dn = torch.max(torch.tensor(1e-36).to(dn.device), dn)
        rddt[lai_sup_0,:] = rdd[lai_sup_0,:] + tdd[lai_sup_0,:] * rsoil[lai_sup_0,:] * tdd[lai_sup_0,:] / dn
        rsdt[lai_sup_0,:] = rsd[lai_sup_0,:] + (tsd[lai_sup_0,:] + tss[lai_sup_0,:]) * rsoil[lai_sup_0,:] * tdd[lai_sup_0,:] / dn
        rdot[lai_sup_0,:] = rdo[lai_sup_0,:] + tdd[lai_sup_0,:] * rsoil[lai_sup_0,:] * (tdo[lai_sup_0,:] + too[lai_sup_0,:]) / dn
        rsodt[lai_sup_0,:] = ((tss[lai_sup_0] + tsd[lai_sup_0,:]) * tdo[lai_sup_0,:] + (tsd[lai_sup_0,:] + tss[lai_sup_0] * rsoil[lai_sup_0,:] * rdd[lai_sup_0,:]) * too[lai_sup_0]) * rsoil[lai_sup_0,:] / dn
        rsost[lai_sup_0,:] = rso + tsstoo[lai_sup_0,:] * rsoil[lai_sup_0,:]
        rsot[lai_sup_0,:] = rsost[lai_sup_0,:] + rsodt[lai_sup_0,:]
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
