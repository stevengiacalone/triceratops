"""Tests for SPOC aperture retrieval.

funcs.get_aperture() reads the SPOC optimal photometric aperture (bit 2
of the aperture bitmask) from the pipeline light-curve FITS file, and
target.get_spoc_apertures() collects one per sector, reporting any it
could not retrieve rather than silently returning a short list.

Network access is mocked, so these run offline.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest


class _FakeHDU:
    def __init__(self, data, header):
        self.data = data
        self.header = header


class _FakeHDUList:
    def __init__(self, aperture):
        self._hdu = _FakeHDU(aperture, {"CRVAL1P": 100, "CRVAL2P": 200})

    def __getitem__(self, key):
        assert key == "APERTURE"
        return self._hdu

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# SPOC aperture bitmask: 1 = collected, 2 = optimal aperture, 4 = centroid
_MASK = np.array([
    [1, 3, 3, 1],
    [1, 7, 7, 5],   # 7 = 1|2|4 (optimal), 5 = 1|4 (not optimal)
    [1, 1, 1, 1],
])


class TestGetAperture:
    def test_extracts_optimal_aperture_bit(self):
        from triceratops.funcs import get_aperture

        with mock.patch("triceratops.funcs.find_url", return_value="x"), \
             mock.patch("triceratops.funcs.fits.open",
                        return_value=_FakeHDUList(_MASK)):
            ap = get_aperture("123", 4)

        # optimal pixels are (row1,col1)=3, (row0,col1)=3, (row0,col2)=3,
        # (row1,col2)=7  -> cols {1,2}+100, rows {0,1}+200
        got = set(map(tuple, ap))
        expected = {(101, 200), (102, 200), (101, 201), (102, 201)}
        assert got == expected

    def test_raises_when_no_optimal_pixels(self):
        from triceratops.funcs import get_aperture

        with mock.patch("triceratops.funcs.find_url", return_value="x"), \
             mock.patch("triceratops.funcs.fits.open",
                        return_value=_FakeHDUList(np.ones((3, 3), int))):
            with pytest.raises(ValueError):
                get_aperture("123", 4)


class TestFindUrlErrors:
    def test_404_raises_file_not_found(self):
        from urllib.error import HTTPError
        from triceratops import funcs

        err = HTTPError("u", 404, "Not Found", {}, None)
        with mock.patch("triceratops.funcs.urlopen", side_effect=err):
            with pytest.raises(FileNotFoundError):
                funcs.find_url("270380593", 4)

    def test_network_error_raises_runtime_error(self):
        from urllib.error import URLError
        from triceratops import funcs

        with mock.patch("triceratops.funcs.urlopen",
                        side_effect=URLError("down")):
            with pytest.raises(RuntimeError):
                funcs.find_url("270380593", 4)


@pytest.mark.heavy
class TestGetSpocApertures:
    @staticmethod
    def _target(mission="TESS", sectors=(4, 5)):
        from triceratops.triceratops import target

        tgt = target.__new__(target)
        tgt.mission = mission
        tgt.ID = 270380593
        tgt.sectors = np.array(sectors)
        return tgt

    def test_non_tess_returns_all_none(self):
        tgt = self._target(mission="Kepler")
        assert tgt.get_spoc_apertures() == [None, None]

    def test_one_entry_per_sector_with_none_for_failures(self):
        tgt = self._target(sectors=(4, 5))

        def fake_get_aperture(ID, sector):
            if sector == 5:
                raise FileNotFoundError("no data")
            return np.array([[1, 2], [3, 4]])

        with mock.patch("triceratops.triceratops.get_aperture",
                        side_effect=fake_get_aperture):
            with pytest.warns(RuntimeWarning):
                aps = tgt.get_spoc_apertures()

        assert len(aps) == 2
        assert aps[0] is not None and aps[1] is None

    def test_all_success_no_warning(self):
        tgt = self._target(sectors=(4,))
        with mock.patch("triceratops.triceratops.get_aperture",
                        return_value=np.array([[1, 2]])):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                aps = tgt.get_spoc_apertures()
        assert len(aps) == 1 and aps[0] is not None


@pytest.mark.heavy
class TestCalcDepthsRejectsNone:
    def test_none_in_apertures_raises(self):
        from triceratops.triceratops import target

        tgt = target.__new__(target)
        tgt.stars = pd.DataFrame({
            "ID": ["T"], "Tmag": [10.0], "mass": [1.0], "rad": [1.0],
            "Teff": [5800.0], "plx": [10.0],
        })
        tgt.pix_coords = [np.array([[2.0, 2.0]]), np.array([[2.0, 2.0]])]
        with pytest.raises(ValueError):
            tgt.calc_depths(
                tdepth=0.01,
                all_ap_pixels=[np.array([[2, 2]]), None],
            )
