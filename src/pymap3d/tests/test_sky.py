from datetime import datetime

import pymap3d as pm
import pymap3d.vallado as pv
import pytest
from pytest import approx

lat, lon = (65, -148)
lla0 = (42, -82, 200)
azel = (180.1, 80)
t0 = datetime(2014, 4, 6, 8)
radec = (166.5032081149338, 55.000011165405752)


def test_azel2radec():
    radec1 = pm.azel2radec(*azel, lat, lon, t0)
    assert radec1 == approx(radec, rel=0.01)
    assert isinstance(radec1[0], float)
    assert isinstance(radec1[1], float)


def test_azel2radec_vallado():
    radec1 = pv.azel2radec(*azel, lat, lon, t0)
    assert radec1 == approx(radec, rel=0.01)
    assert isinstance(radec1[0], float)
    assert isinstance(radec1[1], float)


def test_numpy_azel2radec():
    pytest.importorskip("numpy")
    radec1 = pm.azel2radec([180.1, 180.1], [80, 80], lat, lon, t0)
    assert radec1 == approx(radec, rel=0.01)


def test_radec2azel():
    azel1 = pm.radec2azel(*radec, lat, lon, t0)
    assert azel1 == approx(azel, rel=0.01)
    assert isinstance(azel1[0], float)
    assert isinstance(azel1[1], float)


def test_radec2azel_vallado():
    azel1 = pv.radec2azel(*radec, lat, lon, t0)
    assert azel1 == approx(azel, rel=0.01)
    assert isinstance(azel1[0], float)
    assert isinstance(azel1[1], float)


def test_numpy_radec2azel():
    pytest.importorskip("numpy")
    azel1 = pm.radec2azel([166.503208, 166.503208], [55, 55], lat, lon, t0)
    assert azel1 == approx(azel, rel=0.01)
