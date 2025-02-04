import numpy as np
import torch
import os
import prosail
from pytest import fixture
from distutils import dir_util
from torch.autograd import grad

# n1 = 1.5 
# n2 = 2
# cab = 40.0
# car = 8.0
# cbrown = 0.0
# cw = 0.01
# cm = 0.009 
# lai = 3.0
# lidfa = -0.35
# hspot = 0.01
# tts = 30.0
# tto = 10.0
# psi = 0.0

                     
# input_tensor = torch.tensor([[n1, cab, car, cbrown, cw, cm, lai,
#                              lidfa, hspot, tts, tto, psi],
#                              [n2, cab, car, cbrown, cw, cm, lai,
#                               lidfa, hspot, tts, tto, psi]
#                              ],requires_grad=True)

@fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for locating the test data directory and copying it
    into a temporary directory.
    Taken from  http://www.camillescott.org/2016/07/15/travis-pytest-scipyconf/
    """
    filename = request.module.__file__
    test_dir = os.path.dirname(filename)
    data_dir = os.path.join(test_dir, "data")
    dir_util.copy_tree(data_dir, str(tmpdir))

    def getter(filename, as_str=True):
        filepath = tmpdir.join(filename)
        if as_str:
            return str(filepath)
        return filepath

    return getter


# def test_rsot_prosail5(datadir):
#     fname1 = datadir("REFL_CAN_sim1.txt")
#     fname2 = datadir("REFL_CAN_sim2.txt")
#     w, resv1, rdot1, rsot1, rddt1, rsdt1 = torch.from_numpy(
#         np.loadtxt(fname1, unpack=True)).float()
#     w, resv2, rdot2, rsot2, rddt2, rsdt2 = torch.from_numpy(
#         np.loadtxt(fname2, unpack=True)).float()
#     rr = prosail.run_prosail(input_tensor,
#     typelidf=2,
#     lidfb=torch.as_tensor(-0.15),
#     rsoil=torch.as_tensor(1.0),
#     psoil=torch.as_tensor(1.0),
#     factor="SDR",
#     )
#     assert torch.allclose(torch.vstack([rsot1,rsot2]), rr, atol=0.01)


# def test_rdot_prosail5(datadir):
#     fname1 = datadir("REFL_CAN_sim1.txt")
#     fname2 = datadir("REFL_CAN_sim2.txt")
#     w, resv1, rdot1, rsot1, rddt1, rsdt1 = torch.from_numpy(
#         np.loadtxt(fname1, unpack=True)).float()
#     w, resv2, rdot2, rsot2, rddt2, rsdt2 = torch.from_numpy(
#         np.loadtxt(fname2, unpack=True)).float()
#     rr = prosail.run_prosail(
#         input_tensor,
#         typelidf=2,
#         lidfb=torch.as_tensor(-0.15),
#         rsoil=torch.as_tensor(1.0),
#         psoil=torch.as_tensor(1.0),
#         factor="HDR",
#     )
#     assert torch.allclose(torch.vstack([rdot1,rdot2]), rr, atol=0.01)


# def test_rddt_prosail5(datadir):
#     fname1 = datadir("REFL_CAN_sim1.txt")
#     fname2 = datadir("REFL_CAN_sim2.txt")
#     w, resv1, rdot1, rsot1, rddt1, rsdt1 = torch.from_numpy(
#         np.loadtxt(fname1, unpack=True)).float()
#     w, resv2, rdot2, rsot2, rddt2, rsdt2 = torch.from_numpy(
#         np.loadtxt(fname2, unpack=True)).float()
#     rr = prosail.run_prosail(
#         input_tensor,
#         typelidf=2,
#         lidfb=torch.as_tensor(-0.15),
#         rsoil=torch.as_tensor(1.0),
#         psoil=torch.as_tensor(1.0),
#         factor="BHR",
#     )
#     assert torch.allclose(torch.vstack([rddt1,rddt2]), rr, atol=0.01)


# def test_rsdt_prosail5(datadir):
#     fname1 = datadir("REFL_CAN_sim1.txt")
#     fname2 = datadir("REFL_CAN_sim2.txt")
#     w, resv1, rdot1, rsot1, rddt1, rsdt1 = torch.from_numpy(
#         np.loadtxt(fname1, unpack=True)).float()
#     w, resv2, rdot2, rsot2, rddt2, rsdt2 = torch.from_numpy(
#         np.loadtxt(fname2, unpack=True)).float()
#     rr = prosail.run_prosail(
#         input_tensor,
#         typelidf=2,
#         lidfb=torch.as_tensor(-0.15),
#         rsoil=torch.as_tensor(1.0),
#         psoil=torch.as_tensor(1.0),
#         factor="DHR",
#     )
#     assert torch.allclose(torch.vstack([rsdt1,rsdt2]), rr, atol=0.01)

# def test_grad_prosail5(datadir):
    
#     prosail_grad = grad(prosail.run_prosail(
#         input_tensor,
#         typelidf=2,
#         lidfb=torch.as_tensor(-0.15).float(),
#         rsoil=torch.as_tensor(1.0).float(),
#         psoil=torch.as_tensor(1.0).float(),
#         factor="HDR",
#     ).sum(),input_tensor)[0]
#     assert not prosail_grad.isnan().any()
    
def test_rsdt_prosail5(datadir):
    params_file = datadir("prosail_params.npy")
    rsdt_file = datadir("prosail_rsdt.npy")
    
    tmpdir  = r"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailpython/tests/data/"
    params_file = tmpdir + "prosail_params.npy"
    rsdt_file = tmpdir + "prosail_rsdt.npy"
    params = torch.from_numpy(np.load(params_file)).float()
    prsoil = torch.ones((params.size(0),2))
    params = torch.concat((params, prsoil), axis=1)
    rsdt = torch.from_numpy(np.load(rsdt_file)).float()
    rr = prosail.run_prosail(
        params,
        typelidf=2,
        lidfb=None,
        factor="DHR",
        init_spectra=True
    )
    assert torch.allclose(rsdt, rr, atol=0.01)


def test_rsot_prosail5(datadir):
    params_file = datadir("prosail_params.npy")
    rsot_file = datadir("prosail_rsot.npy")
    params = torch.from_numpy(np.load(params_file)).float()
    prsoil = torch.ones((params.size(0),2))
    params = torch.concat((params, prsoil), axis=1)
    rsot = torch.from_numpy(np.load(rsot_file)).float()
    rr = prosail.run_prosail(
        params,
        typelidf=2,
        lidfb=None,
        factor="SDR",
        init_spectra=True
    )
    assert torch.allclose(rsot, rr, atol=0.01)
    

def test_rddt_prosail5(datadir):
    params_file = datadir("prosail_params.npy")
    rddt_file = datadir("prosail_rddt.npy")
    params = torch.from_numpy(np.load(params_file)).float()
    prsoil = torch.ones((params.size(0),2))
    params = torch.concat((params, prsoil), axis=1)
    rddt = torch.from_numpy(np.load(rddt_file)).float()
    rr = prosail.run_prosail(
        params,
        typelidf=2,
        lidfb=None,
        factor="BHR",
        init_spectra=True
    )
    assert torch.allclose(rddt, rr, atol=0.01)

def test_rdot_prosail5(datadir):
    params_file = datadir("prosail_params.npy")
    rdot_file = datadir("prosail_rdot.npy")
    params = torch.from_numpy(np.load(params_file)).float()
    prsoil = torch.ones((params.size(0),2))
    params = torch.concat((params, prsoil), axis=1)
    rdot = torch.from_numpy(np.load(rdot_file)).float()
    rr = prosail.run_prosail(
        params,
        typelidf=2,
        lidfb=None,
        factor="HDR",
        init_spectra=True
    )
    assert torch.allclose(rdot, rr, atol=0.01)
    
def test_grad_prosail5(datadir):
    params_file = datadir("prosail_params.npy")
    params = torch.from_numpy(np.load(params_file)).float()
    params.requires_grad = True
    prsoil = torch.ones((params.size(0),2))
    params = torch.concat((params, prsoil), axis=1)
    prosail_grad = grad(prosail.run_prosail(
        params,
        typelidf=2,
        lidfb=None,
        factor="HDR",
        init_spectra=True
    ).sum(),params)[0]
    assert not prosail_grad.isnan().any()

tmpdir  = r"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailpython/tests/data/"
params_file = tmpdir + "prosail_params.npy"
rsdt_file = tmpdir + "prosail_rsdt.npy"
params = torch.from_numpy(np.load(params_file)).float()
prsoil = torch.ones((params.size(0),2))
params = torch.concat((params, prsoil), axis=1)
rsdt = torch.from_numpy(np.load(rsdt_file)).float()
rr = prosail.run_prosail(
    params,
    typelidf=2,
    lidfb=None,
    factor="DHR",
    init_spectra=True
)
