#!/usr/bin/env python
"""The PROSPECT leaf optical properties model
Versions 5, D and PRO

Thanks for @jajberni for ProspectPRO implementation!

"""
import torch
from prosail import spectral_lib
from scipy.signal import decimate
from scipy.interpolate import interp1d

def subsample_spectra(tensor:torch.Tensor|None, R_down:int=1, axis:int=0, method:str="block_mean"):
    if tensor is None:
        return None
    if R_down > 1 :
        assert 2100 % R_down == 0
        if method=='block_mean':
            if tensor.size(0)==2101:
                tensor = tensor[:-1].reshape(-1, R_down).mean(1)
        elif method=="decimate":
            device = tensor.device
            decimated_array = decimate(tensor.detach().cpu().numpy(), R_down).copy()
            tensor = torch.from_numpy(decimated_array).to(device)
        elif method == "interp":
            device = tensor.device
            f = interp1d(np.arange(400,2501), tensor.detach().cpu().numpy())
            sampling = np.arange(400, 2501, R_down)
            array = np.apply_along_axis(f, axis, sampling)
            tensor = torch.from_numpy(array).float().to(device)
        else:
            raise NotImplementedError
    return tensor


def init_prospect_spectra(
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
    prospect_version:str="5"):

    if prospect_version == "5":
        nr = spectral_lib.prospect5.nr.to(device) if nr is None else nr
        kab = spectral_lib.prospect5.kab.to(device) if kab is None else kab
        kcar = spectral_lib.prospect5.kcar.to(device) if kcar is None else kcar
        kbrown = spectral_lib.prospect5.kbrown.to(device) if kbrown is None else kbrown
        kw = spectral_lib.prospect5.kw.to(device) if kw is None else kw
        km = spectral_lib.prospect5.km.to(device) if km is None else km
        kant = torch.zeros_like(km)
        kprot = torch.zeros_like(km)
        kcbc = torch.zeros_like(km)

    elif prospect_version.upper() == "D":
        nr = spectral_lib.prospectd.nr.to(device) if nr is None else nr
        kab = spectral_lib.prospectd.kab.to(device) if kab is None else kab
        kcar = spectral_lib.prospectd.kcar.to(device) if kcar is None else kcar
        kbrown = spectral_lib.prospectd.kbrown.to(device) if kbrown is None else kbrown
        kw = spectral_lib.prospectd.kw.to(device) if kw is None else kw
        km = spectral_lib.prospectd.km.to(device) if km is None else km
        kant = spectral_lib.prospectd.kant.to(device) if kant is None else kant
        kprot = torch.zeros_like(km)
        kcbc = torch.zeros_like(km)

    elif prospect_version.upper() == "PRO":
        nr = spectral_lib.prospectpro.nr.to(device) if nr is None else nr
        kab = spectral_lib.prospectpro.kab.to(device) if kab is None else kab
        kcar = spectral_lib.prospectpro.kcar.to(device) if kcar is None else kcar
        kbrown = spectral_lib.prospectpro.kbrown.to(device) if kbrown is None else kbrown
        kw = spectral_lib.prospectpro.kw.to(device) if kw is None else kw
        km = spectral_lib.prospectpro.km.to(device) if km is None else km
        kant = spectral_lib.prospectpro.kant.to(device) if kant is None else kant
        kprot = spectral_lib.prospectpro.kprot.to(device) if kprot is None else kprot
        kcbc = spectral_lib.prospectpro.kcbc.to(device) if kcbc is None else kcbc
    else:
        raise ValueError("prospect_version can only be 5, D or PRO!")
    
    lambdas = torch.arange(400, 2501).float().to(device) if lambdas is None else lambdas
    if R_down > 1 :
        assert 2100 % R_down == 0
        nr = subsample_spectra(nr, R_down=R_down)
        kab = subsample_spectra(kab, R_down=R_down)
        kcar = subsample_spectra(kcar, R_down=R_down)
        kbrown = subsample_spectra(kbrown, R_down=R_down)
        kw = subsample_spectra(kw, R_down=R_down)
        km = subsample_spectra(km, R_down=R_down)
        kant = subsample_spectra(kant, R_down=R_down)
        kprot = subsample_spectra(kprot, R_down=R_down)
        kcbc = subsample_spectra(kcbc, R_down=R_down)
        lambdas = subsample_spectra(lambdas, R_down=R_down)
    return nr, kab, kcar, kbrown, kw, km, kant, kprot, kcbc, lambdas

def pos_expi(x):
    k = torch.arange(1, 75, dtype=x.dtype, device=x.device)
    r = torch.cumprod(x.unsqueeze(-1)*k/torch.square(k+1), dim=-1)
    ga = torch.tensor([0.5772156649015328], dtype=x.dtype, device=x.device)
    y = ga + torch.log(x) + x * (1+(r).sum(-1))
    return y

def ein(x):
    k = torch.arange(1, 75, dtype=x.dtype, device=x.device)
    r = torch.cumprod(- x.unsqueeze(-1)*k/torch.square(k+1), dim=-1)
    return x * (1+(r).sum(-1))


def torch_e1xg(x):
    t0 = torch.zeros_like(x)
    M=20
    for k in range(M,0,-1):
        t0 = k/(1+k/(x+t0))
    e = torch.exp(-x) / (x + t0)
    return e

def torch_e1xl(x):
    ga = torch.tensor([0.5772156649015328], dtype=x.dtype, device=x.device)
    e = - ga - torch.log(x) + ein(x)
    return e

def e1(x,approx_switch=1):
    e = torch.zeros_like(x)
    x_l0 = x<0
    x_la = x<approx_switch
    x_ga = x>=approx_switch
    e[x_l0] = torch.nan
    e[x_la] = torch_e1xl(x[x_la])
    e[x_ga] = torch_e1xg(x[x_ga])
    return e

def neg_expi(x):
    return - e1(-x)

def expi(x):
    ei = torch.zeros_like(x)
    x_l0 = x<0
    x_g0 = x>0
    ei[x_l0] = neg_expi(x[x_l0])
    ei[x_g0] = pos_expi(x[x_g0])
    return ei

class Expi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = expi(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
         i, = ctx.saved_tensors
         return grad_output * (-i.exp()/i)



def run_prospect(
        N, 
        cab, 
        car, 
        cbrown, 
        cw, 
        cm,
        ant=torch.tensor(0.0),
        prot=torch.tensor(0.0),
        cbc=torch.tensor(0.0),
        prospect_version="5",
        nr=None,
        kab=None,
        kcar=None,
        kbrown=None,
        kw=None,
        km=None,
        lambdas=None,
        kant=None,
        kprot=None,
        kcbc=None,
        alpha=torch.as_tensor(40.0),
        device='cpu',
        R_down = 1,
        init_spectra=True
):
    """The PROSPECT model, versions 5, D and PRO.
    This function runs PROSPECT. You can select the version using the
    `prospect_version` argument, and you can also set some of the spectra
    used for model calculations.

    Parameters
    -----------
    n: float
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
    ant: float, optional
        Anthocyanins content. Used in Prospect-D and Prospect-PRO [g cm^{-2}]
    prot: float, optional
        Protein content. Used in Prospect-PRO. [g cm^{-2}]
    cbc: float, optional
        Carbon based constituents. Used in Prospect-PRO. [g cm^{-2}]
    prospect_version: string, optiona, default "D".
        The version of PROSPECT, "5", "D" or "PRO".
    nr: array, optional
        The refractive index of the leaf. If `None` (default), will use the
        values for the selected PROPSECT version. [-].
    kab: 2101-element array, optional
        The specific absorption coefficient of chlorophyll (a+b) [cm^2 ug^{-1}].
    kcar: 2101-element array, optional
        The specific absorption coefficient of carotenoids [cm^2 ug^{-1}].
    kbrown:  2101-element array, optional
        The specific absorption coefficient of brown pigments (arbitrary units).
    kw:  2101-element array, optional
        The specific absorption coefficient of water (cm^{-1}).
    km: 2101-element array, optional
        The specific absorption coefficient of dry matter [cm^2 g^{-1}].
    kant: 2101-element array, optional
        The specific absorption coefficient of Anthocyanins [cm^2 nmol^{-1}].
    kprot: 2101-element array, optional
        The specific absorption coefficient of proteins [cm^2 g^{-1}].
    kcbc: 2101-element array, optional
        The specific absorption coefficient of carbon based constituents [cm^2 ug^{-1}].
    alpha: float, optional, default 40..
        Maximum incident angle relative to the normal of the leaf plane. [deg]


    Returns
    -------

    3 arrays of the size 2101: the wavelengths in [nm], the leaf reflectance
    and transmittance.

    """
    if init_spectra:
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
                                         R_down=R_down, prospect_version=prospect_version)
    
    
    wv, refl, trans = prospect_d(
        N, 
        cab, 
        car, 
        cbrown, 
        cw, 
        cm, 
        ant, 
        prot, 
        cbc,
        nr,
        kab,
        kcar,
        kbrown,
        kw,
        km,
        kant,
        kprot,
        kcbc,
        alpha=alpha,
        lambdas=lambdas
    )

    return wv, refl, trans


def calctav(alpha, nr):
    # ***********************************************************************
    # calctav
    # ***********************************************************************
    # Stern F. (1964), Transmission of isotropic radiation across an
    # interface between two dielectrics, Appl. Opt., 3(1):111-113.
    # Allen W.A. (1973), Transmission of isotropic light across a
    # dielectric surface in two and three dimensions, J. Opt. Soc. Am.,
    # 63(6):664-666.
    # ***********************************************************************

    # rd  = pi/180 torch.deg2rad
    n2 = nr * nr
    npx = n2 + 1
    nm = n2 - 1
    a = (nr + 1) * (nr + 1) / 2.0
    k = -(n2 - 1) * (n2 - 1) / 4.0
    sa = torch.sin(torch.deg2rad(alpha))

    if alpha != 90:
        b1 = torch.sqrt((sa * sa - npx / 2) * (sa * sa - npx / 2) + k)
    else:
        b1 = 0.0
    b2 = sa * sa - npx / 2
    b = b1 - b2
    b3 = b**3
    a3 = a**3
    ts = (k**2 / (6 * b3) + k / b - b / 2) - (k**2.0 /
                                              (6 * a3) + k / a - a / 2)

    tp1 = -2 * n2 * (b - a) / (npx**2)
    tp2 = -2 * n2 * npx * torch.log(b / a) / (nm**2)
    tp3 = n2 * (1 / b - 1 / a) / 2
    tp4 = (16 * n2**2 * (n2**2 + 1) * torch.log(
        (2 * npx * b - nm**2) / (2 * npx * a - nm**2)) / (npx**3 * nm**2))
    tp5 = (16 * n2**3 * (1.0 / (2 * npx * b - nm**2) - 1 /
                         (2 * npx * a - nm**2)) / (npx**3))
    tp = tp1 + tp2 + tp3 + tp4 + tp5
    tav = (ts + tp) / (2 * sa**2)

    return tav


def refl_trans_one_layer(alpha, nr, tau):
    # ***********************************************************************
    # reflectance and transmittance of one layer
    # ***********************************************************************
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    # ***********************************************************************
    # reflectivity and transmissivity at the interface
    # -------------------------------------------------
    talf = calctav(alpha, nr)
    ralf = 1.0 - talf
    t12 = calctav(torch.as_tensor(90.0), nr)
    r12 = 1.0 - t12
    t21 = t12 / (nr * nr)
    r21 = 1 - t21

    # top surface side
    denom = 1.0 - r21 * r21 * tau * tau
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta

    # bottom surface side
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    return r, t, Ra, Ta, denom


def prospect_d(
        N, 
        cab, 
        car, 
        cbrown, 
        cw, 
        cm,
        ant,
        prot,
        cbc,
        nr,
        kab,
        kcar,
        kbrown,
        kw,
        km,
        kant,
        kprot,
        kcbc,
        alpha=torch.as_tensor(40.0),
        R_down=1,
        lambdas=None):
        
    # wavelengths
    if lambdas is None:
        if R_down > 1:
            lambdas = torch.arange(400, 2500, R_down).float().to(N.device)
        else:
            lambdas = torch.arange(400, 2501).float().to(N.device)
    n_lambdas = len(lambdas)
    #TODO:  Perform only at intialization
    n_elems_list = [
        len(spectrum)
        for spectrum in [nr, kab, kcar, kbrown, kw, km, kant, kprot, kcbc]
    ]
    if not all(n_elems == n_lambdas for n_elems in n_elems_list):
        raise ValueError("Leaf spectra don't have the right shape!")

    kall = (cab * kab + car * kcar + ant * kant + cbrown * kbrown + cw * kw +
            cm * km + prot * kprot + cbc * kcbc) / N

    j = kall > 0
    t1 = (1 - kall) * torch.exp(-kall)
    t2 = kall**2 * (-Expi.apply(-kall))
    tau = torch.ones_like(t1)
    tau[j] = t1[j] + t2[j]

    r, t, Ra, Ta, denom = refl_trans_one_layer(alpha, nr, tau)

    # ***********************************************************************
    # reflectance and transmittance of N layers
    # Stokes equations to compute properties of next N-1 layers (N real)
    # Normal case
    # ***********************************************************************
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    # ***********************************************************************
    D = torch.sqrt((1 + r + t) * (1 + r - t) * (1.0 - r + t) * (1.0 - r - t))
    rq = r * r
    tq = t * t
    a = (1 + rq - tq + D) / (2 * r)
    b = (1 - rq + tq + D) / (2 * t)

    bNm1 = torch.pow(b, N - 1)
    bN2 = bNm1 * bNm1
    a2 = a * a
    denom = a2 * bN2 - 1
    Rsub = a * (bN2 - 1) / denom
    Tsub = bNm1 * (a2 - 1) / denom

    # Case of zero absorption
    j = r + t >= 1.0
    if len(torch.where(j)[0])>0:
        Tsub[j] = t[j] / (t[j] + (1 - t[j]) * (N[torch.where(j)[0]].squeeze() - 1))
        Rsub[j] = 1 - Tsub[j]

    # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom = 1 - Rsub * r
    tran = Ta * Tsub / denom
    refl = Ra + Ta * Rsub * t / denom

    return lambdas, refl, tran


###if __name__ == "__main__":

###k_cab = prosail.spectral_libs.k_cab
###k_w = prosail.spectral_libs.k_cw
###k_m = prosail.spectral_libs.k_cm
###k_car = prosail.spectral_libs.k_car
###k_brown = prosail.spectral_libs.k_brown
###nr = prosail.spectral_libs.refractive

###wv, r, t = prospect_d (2.1, 60., 10., 0.1, 0.013, 0.016, 0,
###nr, k_cab, k_car, k_brown, k_w, k_m, k_m*0.,
###alpha=40.)

###rt = prosail.prospect_5b(2.1, 60., 10., 0.1, 0.013, 0.016)
###plt.plot(wv, r-rt[:,0], '--')
###plt.plot(wv, t-rt[:,1], '--')

###wv, r, t = prospect_d (2.1, 10., 10., 0.1, 0.013, 0.016, 0,
###nr, k_cab, k_car, k_brown, k_w, k_m, k_m*0.,
###alpha=40.)
####    plt.plot(wv, r)
###rt = prosail.prospect_5b(2.1, 10., 10., 0.1, 0.013, 0.016)
###plt.plot(wv, r-rt[:,0], '-')
###plt.plot(wv, t-rt[:,1], '-')
