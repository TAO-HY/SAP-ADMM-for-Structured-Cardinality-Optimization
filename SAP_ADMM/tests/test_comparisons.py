"""Comparison algorithms: independent proximal/variational checks and workflow QA."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from sap_admm import half_threshold, padmm_l0, sdcam, sap_admm_l1_image
from sap_admm.comparisons import moreau_objective
from sap_admm.utils import difference_matrix, image_operators
from experiments._shared import load_config
from experiments import plot_signal, plot_mnist, run_signal, run_mnist


@pytest.mark.parametrize("weight", [0.001, 0.1, 1.0, 5.0])
def test_half_prox_is_global_scalar_minimizer(weight):
    threshold = 1.5*weight**(2/3)
    values = np.r_[0, threshold*np.array([0.5, 0.999, 1.0, 1.001, 2, 10])]
    result = half_threshold(values, weight)
    for v,u in zip(values,result):
        # Independent minimization after u=a^2 removes the derivative singularity.
        objective = lambda a: .5*(a*a-v)**2 + weight*a
        points = np.linspace(0, np.sqrt(max(v,1)), 1001)
        j = np.argmin([objective(a) for a in points])
        opt = minimize_scalar(objective, bounds=(points[max(0,j-1)], points[min(1000,j+1)]), method="bounded",
                              options={"xatol": 1e-13})
        minimum = min(objective(0), opt.fun)
        assert .5*(u-v)**2 + weight*np.sqrt(abs(u)) == pytest.approx(minimum, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(half_threshold(-values,weight),-result)


@pytest.mark.parametrize("loss", ["l1", "l2"])
def test_moreau_dc_direction_matches_numerical_derivative(loss):
    b=np.array([.1,.2,-.1,.3])
    x=np.array([0.,2.,-.8,.9])
    matrix=difference_matrix(4)
    D=lambda v:matrix@v
    mu,weight=.2,.05
    value,_,dx,prox=moreau_objective(x,b,D,weight,mu,loss)
    expected=matrix.T@(dx-prox)/mu+(np.sign(x-b)/4 if loss=="l1" else x-b)
    numerical=[]
    for k in range(4):
        e=np.zeros(4);e[k]=1e-6
        numerical.append((moreau_objective(x+e,b,D,weight,mu,loss)[0]-moreau_objective(x-e,b,D,weight,mu,loss)[0])/2e-6)
    np.testing.assert_allclose(expected,numerical,rtol=1e-6,atol=1e-7)


def test_padmm_restoration_matches_small_exact_segmentation():
    b=np.array([0.,.1,-.1,3.,3.1,2.9])
    weight=1.0
    best=(float("inf"),None)
    for cuts in itertools.product((False,True),repeat=len(b)-1):
        boundaries=[0]+[j+1 for j,v in enumerate(cuts) if v]+[len(b)]
        x=np.empty_like(b)
        for a,c in zip(boundaries,boundaries[1:]):x[a:c]=b[a:c].mean()
        cost=.5*np.sum((x-b)**2)+weight*sum(cuts)
        if cost<best[0]:best=(cost,x)
    x,y,info=padmm_l0(b,difference_matrix(len(b)),{"lambda_l0":weight,"stop_tol":1e-8})
    np.testing.assert_allclose(x,best[1],atol=1e-6)
    assert info["constraint_residual"] < 1e-7
    assert info["objective"] == pytest.approx(best[0],abs=1e-6)


@pytest.mark.parametrize("loss", ["l1", "l2"])
def test_sdcam_continuation_and_cap_are_reported(loss):
    b=np.array([0.,.1,-.1,3.,3.1,2.9])
    matrix=difference_matrix(len(b))
    x,y,info=sdcam(b,(lambda v:matrix@v,lambda v:matrix.T@v),{"max_inner":7},loss=loss)
    assert len(info["stages"])==8
    assert info["smoothing_final"] == pytest.approx(1e-8)
    assert info["smoothing_next"] < 1e-8
    assert info["iter"] == sum(v["iterations"] for v in info["stages"])
    assert all(v["iterations"]<=7 for v in info["stages"])
    assert all(v["epsilon"]>=1e-6 for v in info["stages"])
    np.testing.assert_allclose(y,matrix@x)
    assert np.isfinite(x).all() and not info["nu_applicable"]


def test_sdcam_does_not_commit_a_failed_line_search():
    b=np.array([0.,2.,-1.,3.]);matrix=difference_matrix(4)
    with pytest.raises(RuntimeError,match="no step was committed"):
        sdcam(b,(lambda v:matrix@v,lambda v:matrix.T@v),
              {"Lmin":1e-8,"Lmax":1e-8,"max_backtracks":1},loss="l2")


def test_supplement_l1_image_checks_auxiliary_blocks():
    _,_,info=sap_admm_l1_image(np.arange(12).reshape(3,4)/12,parameters={"max_iter":2,"stop_tol":1e-20})
    assert info["stopping_profile"]=="supplement"
    assert info["stopping_blocks"]=="x,y,p,q,eta,mu"
    assert info["effective_max_iter"]==2


def test_comparison_updates_match_saved_reference_fixture():
    folder=Path(__file__).parent/"fixtures"
    with np.load(folder/"comparison_reference_outputs.npz",allow_pickle=False) as data:
        b=data["signal_input"];matrix=difference_matrix(len(b))
        for loss in ("l1","l2"):
            x,_,info=sdcam(b,(lambda v:matrix@v,lambda v:matrix.T@v),
                          {"lambda_half":.05 if loss=="l1" else 5.,"lambda_init":.1,
                           "lambda_min":.1,"max_inner":8,"epsilon_init":1e-30,"relative_objective_tol":1e-30},loss=loss)
            np.testing.assert_allclose(x,data[f"sdcam_signal_{loss}"],atol=2e-12,rtol=2e-12)
        x,y,_=padmm_l0(b,matrix,{"max_iter":40,"stop_tol":1e-30})
        np.testing.assert_allclose(x,data["padmm_x"],atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(y,data["padmm_y"],atol=2e-12,rtol=2e-12)


def test_all_method_figures_preserve_stored_inputs(tmp_path,monkeypatch):
    captured={}
    for m in (plot_signal,plot_mnist):
        monkeypatch.setattr(m,"export",lambda f,p,n,**kw:captured.update({n:f}))
    config=load_config("signal.json");config.update(n=12,trials=1,probabilities=[0.,.2],max_iter=5)
    config["sdcam"]["max_inner"]=5;config["padmm"]["max_iter"]=5
    run_signal.run(config,tmp_path/"signal");plot_signal.plot(tmp_path/"signal")
    assert len(captured["F1_Time_Efficiency"].axes[0].lines)==6
    assert len(captured["F1_Distribution"].axes)==4
    assert len(captured["Figure2_Recovery_Comparison_FiveModels_acc_capl1_python"].axes)==12
    with np.load(tmp_path/"signal"/"results.npz") as data:
        methods=list(data["methods"])
        for method,line in zip(("sap_admm", "sap_admm_halpern", "sdcam_l1", "sap_admm_l1", "sdcam_l2", "padmm_l0"),captured["F1_Time_Efficiency"].axes[0].lines):
            np.testing.assert_allclose(line.get_ydata(),data["f1"].mean(0)[:,methods.index(method)])
    records=json.loads((tmp_path/"signal"/"solver_diagnostics.json").read_text())
    assert len(records)==12 and all("stages" in r["solver"] for r in records if r["Method"].startswith("sdcam"))
    config=load_config("mnist.json");config.update(trials=1,digits=[0,1]);config["admm"]["max_iter"]=5;config["sdcam"]["max_inner"]=5
    run_mnist.run(config,tmp_path/"mnist");plot_mnist.plot(tmp_path/"mnist")
    assert len(captured["mnist_four_model_restoration_python"].axes)==12
    import matplotlib.pyplot as plt
    for figure in captured.values():plt.close(figure)
