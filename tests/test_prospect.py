import numpy as np
import torch
import os
import prosail
from distutils import dir_util
from pytest import fixture
from torch.autograd import grad

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

def test_refl_prospect5(datadir):
    params_file = datadir("prospect5_params.npy")
    refl_file = datadir("prospect5_refl.npy")
    input_tensor = torch.from_numpy(np.load(params_file)).float()
    true_refl = torch.from_numpy(np.load(refl_file)).float()
    N = input_tensor[:,0].reshape(-1,1)
    cab = input_tensor[:,1].reshape(-1,1)
    car = input_tensor[:,2].reshape(-1,1)
    cbrown = input_tensor[:,3].reshape(-1,1)
    cw = input_tensor[:,4].reshape(-1,1)
    cm = input_tensor[:,5].reshape(-1,1)
    w, refl, trans  = prosail.run_prospect(N=N, 
					cab=cab, 
					car=car, 
					cbrown=cbrown, 
					cw=cw, 
					cm=cm, 
					prospect_version="5", init_spectra=True)
    assert torch.allclose(true_refl, refl, atol=0.01)

def test_grad_prospect5(datadir):
    params_file = datadir("prospect5_params.npy")
    input_tensor = torch.from_numpy(np.load(params_file)).float()
    input_tensor.requires_grad = True
    N = input_tensor[:,0].reshape(-1,1)
    cab = input_tensor[:,1].reshape(-1,1)
    car = input_tensor[:,2].reshape(-1,1)
    cbrown = input_tensor[:,3].reshape(-1,1)
    cw = input_tensor[:,4].reshape(-1,1)
    cm = input_tensor[:,5].reshape(-1,1)
    w, refl, trans  = prosail.run_prospect(N=N, 
					cab=cab, 
					car=car, 
					cbrown=cbrown, 
					cw=cw, 
					cm=cm, 
    					prospect_version="5", init_spectra=True)
    prospect_grad = grad(refl.sum(), input_tensor)[0]
    assert not prospect_grad.isnan().any()
