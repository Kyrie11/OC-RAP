from __future__ import annotations
import os
import numpy as np
import pytest

from ocrap.v48_74_signed_viability import (
    V48_74_ENV,
    apply_v48_74_feature_overlay,
    signed_viability_diagnostics,
)


def test_safe_separating_trace_has_no_first_order_debt():
    t=np.array([0.,1.,2.,3.])
    h=np.array([2.,2.5,3.,3.5])
    d=signed_viability_diagnostics(h,t)
    assert float(d.first_order_debt)==0.0
    assert float(d.first_order_factor)==1.0


def test_contact_without_escape_has_positive_debt():
    t=np.array([0.,1.,2.,3.])
    h=np.array([-1.,-1.,-1.,-1.])
    d=signed_viability_diagnostics(h,t,clearance_scale=1.0)
    assert float(d.first_order_debt)>0.0
    assert float(d.second_order_debt)>0.0


def test_contact_with_linear_finite_time_escape_is_first_order_viable():
    t=np.array([0.,1.,2.,3.])
    h=np.array([-3.,-2.,-1.,0.])
    d=signed_viability_diagnostics(h,t,clearance_scale=1.0)
    assert np.isclose(float(d.first_order_debt),0.0,atol=1e-10)


def test_near_contact_closing_is_penalized():
    t=np.array([0.,1.,2.,3.])
    h=np.array([0.4,0.3,0.2,0.1])
    d=signed_viability_diagnostics(h,t,clearance_scale=1.0)
    assert float(d.first_order_debt)>0.0


def test_overlay_is_identity_when_disabled(monkeypatch):
    monkeypatch.delenv(V48_74_ENV,raising=False)
    x=np.arange(44,dtype=np.float64).reshape(2,22)
    y=apply_v48_74_feature_overlay(x,{"signed_clearance":np.array([1.,1.,1.]),"times_s":np.arange(3.)})
    assert y is x


def test_overlay_replaces_only_last_two_coordinates(monkeypatch):
    monkeypatch.setenv(V48_74_ENV,"1")
    x=np.arange(44,dtype=np.float64).reshape(2,22)
    h=np.stack([np.array([-1.,-.5,0.]),np.array([1.,.5,.1])])
    y=apply_v48_74_feature_overlay(x,{"projected_recovery_signed_clearance":h,"future_times_s":np.arange(3.),"clearance_scale":np.ones(2)})
    np.testing.assert_array_equal(y[...,:20],x[...,:20])
    assert np.all(np.isfinite(y[...,20:]))
    assert not np.array_equal(y[...,20:],x[...,20:])


def test_overlay_fails_closed_without_time_resolved_clearance(monkeypatch):
    monkeypatch.setenv(V48_74_ENV,"1")
    x=np.zeros((2,22),dtype=np.float64)
    with pytest.raises(RuntimeError):
        apply_v48_74_feature_overlay(x,{"clearance":0.1,"times_s":np.arange(3.)})
