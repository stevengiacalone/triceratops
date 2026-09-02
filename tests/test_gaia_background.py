"""Tests for the Gaia DR3 background-population option.

query_gaia_background() pulls real Gaia sources for the blended-star
(DTP/DEB/BTP/BEB) scenarios in place of a TRILEGAL simulation, writing a
CSV in the same format trilegal_results() reads.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from triceratops.funcs import gaia_to_Tmag, trilegal_results


class TestGaiaToTmag:
    def test_solar_colour_offset(self):
        # BP-RP ~ 0.82 (Sun): T - G ~ -0.43
        assert float(gaia_to_Tmag(10.0, 10.41, 9.59)) - 10.0 == \
            pytest.approx(-0.43, abs=0.02)

    def test_red_star_is_brighter_in_T(self):
        # an M dwarf (BP-RP ~ 2.5) is ~1 mag brighter in T than in G
        assert float(gaia_to_Tmag(12.0, 13.25, 10.75)) < 11.2

    def test_missing_colour_uses_mean_offset(self):
        assert float(gaia_to_Tmag(15.0, np.nan, np.nan)) == \
            pytest.approx(15.0 - 0.43)

    def test_vectorized(self):
        out = gaia_to_Tmag(np.array([10.0, 12.0]),
                           np.array([10.4, 13.0]),
                           np.array([9.6, 11.0]))
        assert out.shape == (2,)


class TestTrilegalResultsFileFormats:
    @staticmethod
    def _clean_frame():
        return pd.DataFrame({
            "Mact": [1.0, 0.5, 0.3],
            "logg": [4.4, 4.8, 5.0],
            "logTe": [3.76, 3.55, 3.50],
            "[M/H]": [0.0, 0.0, 0.0],
            "TESS": [15.0, 18.0, 19.5],
            "J": [14.0, 17.0, 18.2],
            "H": [13.5, 16.5, 17.6],
            "Ks": [13.4, 16.4, 17.5],
        })

    def test_reads_gaia_style_csv_no_footer(self, tmp_path):
        f = tmp_path / "gaia.csv"
        self._clean_frame().to_csv(f, index=False)
        Tmags, masses, loggs, Teffs, Zs, J, H, K = trilegal_results(str(f), 0.0)
        assert len(Tmags) == 3
        assert np.all(np.isfinite(masses))
        assert Teffs[0] == pytest.approx(10**3.76, rel=1e-6)

    def test_strips_trilegal_footer_rows(self, tmp_path):
        df = self._clean_frame()
        df.loc[3] = ["#TRILEGAL", "normally", np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan]
        df.loc[4] = ["terminated", np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan]
        f = tmp_path / "trilegal.csv"
        df.to_csv(f)  # with index, like save_trilegal
        Tmags = trilegal_results(str(f), 0.0)[0]
        assert len(Tmags) == 3  # the two footer rows dropped

    def test_faint_cut_keeps_only_fainter_than_target(self, tmp_path):
        f = tmp_path / "g.csv"
        self._clean_frame().to_csv(f, index=False)
        Tmags = trilegal_results(str(f), 17.0)[0]
        assert set(np.round(Tmags, 1)) == {18.0, 19.5}

    def test_derives_T_from_JK_when_no_TESS_column(self, tmp_path):
        df = self._clean_frame().drop(columns=["TESS"])
        f = tmp_path / "no_tess.csv"
        df.to_csv(f, index=False)
        Tmags = trilegal_results(str(f), 0.0)[0]
        assert len(Tmags) == 3
        assert np.all(np.isfinite(Tmags))


@pytest.mark.heavy
class TestQueryGaiaBackground:
    @staticmethod
    def _fake_gaia_table(n=50, seed=0):
        rng = np.random.default_rng(seed)
        from astropy.table import Table
        G = rng.uniform(12, 21, n)
        return Table({
            "phot_g_mean_mag": G,
            "phot_bp_mean_mag": G + rng.uniform(0.3, 2.0, n),
            "phot_rp_mean_mag": G - rng.uniform(0.3, 1.5, n),
            "parallax": rng.uniform(-0.5, 5.0, n),
            "parallax_over_error": rng.uniform(0.0, 20.0, n),
        })

    def test_writes_expected_columns(self, tmp_path, monkeypatch):
        from triceratops import funcs

        monkeypatch.chdir(tmp_path)
        fake = self._fake_gaia_table()
        with mock.patch("astroquery.gaia.Gaia") as G:
            G.launch_job.return_value.get_results.return_value = fake
            fname = funcs.query_gaia_background(10.0, -20.0, 42, verbose=0)

        assert fname is not None
        df = pd.read_csv(tmp_path / fname)
        assert list(df.columns) == [
            "Mact", "logg", "logTe", "[M/H]", "TESS", "J", "H", "Ks"
        ]
        assert len(df) > 0
        assert df.notna().all().all()
        # feeds trilegal_results cleanly
        out = trilegal_results(str(tmp_path / fname), 0.0)
        assert all(np.all(np.isfinite(a)) for a in out)

    def test_returns_none_on_query_failure(self, tmp_path, monkeypatch):
        from triceratops import funcs

        monkeypatch.chdir(tmp_path)
        with mock.patch("astroquery.gaia.Gaia") as G:
            G.launch_job.side_effect = RuntimeError("archive down")
            G.launch_job_async.side_effect = RuntimeError("archive down")
            assert funcs.query_gaia_background(10.0, -20.0, 42, verbose=0) is None

    def test_returns_none_when_field_empty(self, tmp_path, monkeypatch):
        from triceratops import funcs
        from astropy.table import Table

        monkeypatch.chdir(tmp_path)
        with mock.patch("astroquery.gaia.Gaia") as G:
            G.launch_job.return_value.get_results.return_value = Table(
                {"phot_g_mean_mag": [], "phot_bp_mean_mag": [],
                 "phot_rp_mean_mag": [], "parallax": [],
                 "parallax_over_error": []}
            )
            assert funcs.query_gaia_background(10.0, -20.0, 42, verbose=0) is None
