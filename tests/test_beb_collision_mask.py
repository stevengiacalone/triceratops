"""Tests for NC-04: EB collision mask fixes.

Bug 1 (lnZ_BEB): The parallel code path used coll_twin (twin-period collision
mask) for the q < 0.95 branch instead of coll (standard-period collision mask).
Since a_twin > a (Kepler's 3rd law), coll_twin is less restrictive, admitting
physically impossible configurations.

Bug 2 (lnZ_NEB_unknown): The parallel code path used coll (standard-period
collision mask) for the q >= 0.95 twin branch instead of coll_twin. This is
the inverse of Bug 1: there the standard branch used the twin mask; here the
twin branch uses the standard mask.

Both serial code paths were already correct. These fixes align the parallel
paths with their serial counterparts.
"""

import inspect
import re
import textwrap

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mask-logic tests (no heavy imports required)
# ---------------------------------------------------------------------------

def _build_test_arrays(n=200, seed=42):
    """Build synthetic arrays that mimic lnZ_BEB's mask inputs.

    Returns a dict with:
        qs       - mass ratios, ~uniform [0, 1]
        incs     - inclinations (all 90 deg, i.e. edge-on, so inc >= inc_min)
        inc_min  - minimum inclination (all 0 deg, so everything transits)
        coll     - standard-period collision flag
        coll_twin - twin-period collision flag
        loggs    - companion log(g), all 4.0 (passes >= 3.5 cut)
        Teffs    - companion Teff, all 5000 (passes <= 10000 cut)
    """
    rng = np.random.default_rng(seed)
    qs = rng.uniform(0, 1, n)
    return dict(
        qs=qs,
        incs=np.full(n, 90.0),
        inc_min=np.full(n, 0.0),
        coll=np.ones(n, dtype=bool),           # all colliding at standard period
        coll_twin=np.zeros(n, dtype=bool),      # none colliding at twin period
        loggs=np.full(n, 4.0),
        Teffs=np.full(n, 5000.0),
    )


def _parallel_mask_q_lt_95(incs, inc_min, coll, coll_twin, qs, loggs, Teffs):
    """Reproduce the CORRECTED parallel mask for the q < 0.95 branch.

    This matches the fixed line 2298-2304 of marginal_likelihoods.py:
        mask = (
            (incs >= inc_min)
            & (coll == False)         # <-- NC-04 fix: was coll_twin
            & (qs < 0.95)
            & (loggs >= 3.5)
            & (Teffs <= 10000)
        )
    """
    return (
        (incs >= inc_min)
        & (coll == False)
        & (qs < 0.95)
        & (loggs >= 3.5)
        & (Teffs <= 10000)
    )


def _buggy_mask_q_lt_95(incs, inc_min, coll, coll_twin, qs, loggs, Teffs):
    """Reproduce the BUGGY parallel mask (before the fix).

    Uses coll_twin instead of coll for the q < 0.95 branch.
    """
    return (
        (incs >= inc_min)
        & (coll_twin == False)
        & (qs < 0.95)
        & (loggs >= 3.5)
        & (Teffs <= 10000)
    )


def _serial_mask_q_lt_95_i(incs_i, inc_min, coll_i, qs_i, loggs_i, Teffs_i):
    """Reproduce the serial mask for a single draw (q < 0.95), already correct."""
    return (
        (incs_i >= inc_min)
        & (qs_i < 0.95)
        & (loggs_i >= 3.5)
        & (Teffs_i <= 10000)
        & (coll_i == False)
    )


def _parallel_mask_q_ge_95(incs, inc_min, coll, coll_twin, qs, loggs, Teffs):
    """Reproduce the parallel mask for the q >= 0.95 (twin) branch.

    This correctly uses coll_twin in both the parallel and serial paths.
    """
    return (
        (incs >= inc_min)
        & (coll_twin == False)
        & (qs >= 0.95)
        & (loggs >= 3.5)
        & (Teffs <= 10000)
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestBEBCollisionMask:
    """NC-04: The q < 0.95 branch must use coll, not coll_twin."""

    def test_corrected_mask_blocks_collisions(self):
        """When coll=True for all draws, the corrected mask excludes all q<0.95."""
        d = _build_test_arrays()
        mask = _parallel_mask_q_lt_95(**d)

        q_lt_95 = d["qs"] < 0.95
        assert np.any(q_lt_95), "test setup: need some q < 0.95 draws"
        # coll is all True -> corrected mask should block every q < 0.95 draw
        assert not np.any(mask), (
            "corrected mask must exclude all draws when coll=True"
        )

    def test_buggy_mask_admits_collisions(self):
        """The buggy mask (coll_twin=False) incorrectly admits q<0.95 draws."""
        d = _build_test_arrays()
        mask_buggy = _buggy_mask_q_lt_95(**d)

        q_lt_95 = d["qs"] < 0.95
        # coll_twin is all False -> buggy mask would pass q < 0.95 draws
        admitted = mask_buggy & q_lt_95
        assert np.any(admitted), (
            "test setup: buggy mask should admit some q<0.95 draws "
            "(coll_twin=False does not block them)"
        )

    def test_corrected_more_restrictive_than_buggy(self):
        """The corrected mask is strictly more restrictive than the buggy one.

        Every draw admitted by the corrected mask must also be admitted by the
        buggy mask, but not vice versa (when coll and coll_twin diverge).
        """
        d = _build_test_arrays()
        mask_correct = _parallel_mask_q_lt_95(**d)
        mask_buggy = _buggy_mask_q_lt_95(**d)

        # Correct is a subset of buggy (since coll=True is stricter than coll_twin=False)
        assert np.all(mask_correct <= mask_buggy), (
            "corrected mask must be at least as restrictive as buggy mask"
        )
        # And strictly more restrictive in this case
        assert np.sum(mask_buggy) > np.sum(mask_correct), (
            "corrected mask should block more draws when coll and coll_twin diverge"
        )

    def test_parallel_serial_symmetry_q_lt_95(self):
        """After the fix, parallel and serial masks agree for q < 0.95."""
        d = _build_test_arrays()
        mask_parallel = _parallel_mask_q_lt_95(**d)

        # Build the serial mask element-by-element
        n = len(d["qs"])
        mask_serial = np.zeros(n, dtype=bool)
        for i in range(n):
            if d["qs"][i] < 0.95:
                mask_serial[i] = _serial_mask_q_lt_95_i(
                    d["incs"][i], d["inc_min"][i],
                    d["coll"][i], d["qs"][i],
                    d["loggs"][i], d["Teffs"][i],
                )

        np.testing.assert_array_equal(
            mask_parallel, mask_serial,
            err_msg="parallel and serial masks must agree for q < 0.95",
        )

    def test_twin_branch_still_uses_coll_twin(self):
        """The q >= 0.95 branch must still use coll_twin (no regression)."""
        d = _build_test_arrays()
        # Override: coll=True, coll_twin=False (same as default from _build_test_arrays)
        mask_twin = _parallel_mask_q_ge_95(**d)

        q_ge_95 = d["qs"] >= 0.95
        if np.any(q_ge_95):
            # coll_twin=False should NOT block q >= 0.95 draws
            assert np.any(mask_twin), (
                "q>=0.95 branch should admit draws when coll_twin=False"
            )

    def test_twin_branch_unaffected_by_coll(self):
        """The twin branch does not depend on coll at all."""
        d = _build_test_arrays()
        mask_with_coll_true = _parallel_mask_q_ge_95(**d)

        # Flip coll to all False -- twin branch should be identical
        d2 = dict(d, coll=np.zeros(len(d["qs"]), dtype=bool))
        mask_with_coll_false = _parallel_mask_q_ge_95(**d2)

        np.testing.assert_array_equal(
            mask_with_coll_true, mask_with_coll_false,
            err_msg="twin branch mask must not depend on coll",
        )


# ---------------------------------------------------------------------------
# Source-level regression test
# ---------------------------------------------------------------------------

class TestSourceRegression:
    """Verify the actual source line uses the correct collision variable.

    Rather than parsing the AST of the monolithic lnZ_BEB function, we
    read the source file directly and inspect the specific mask blocks
    in the parallel code path (the only place the bug existed).
    """

    @staticmethod
    def _read_lnZ_BEB_source():
        """Return the full source of lnZ_BEB as a string."""
        from triceratops.marginal_likelihoods import lnZ_BEB
        return inspect.getsource(lnZ_BEB)

    @staticmethod
    def _extract_parallel_mask_blocks(source):
        """Extract multi-line mask blocks from the parallel path of lnZ_BEB.

        These are the mask blocks that appear after 'if parallel:' and
        use the multi-line format:
            mask = (
                ...
                )
        """
        # Find all multi-line mask blocks (mask = (\n ...\n  ))
        # The closing ) sits on its own indented line.
        blocks = list(re.finditer(
            r"mask\s*=\s*\(.*?\n\s*\)",
            source,
            re.DOTALL,
        ))
        # Filter to only those containing qs comparisons (BEB-specific)
        beb_blocks = [b for b in blocks if "qs" in b.group()]
        return beb_blocks

    def test_parallel_q_lt_95_uses_coll_not_coll_twin(self):
        """The parallel q < 0.95 mask must use (coll == False)."""
        source = self._read_lnZ_BEB_source()
        blocks = self._extract_parallel_mask_blocks(source)

        # Find the q < 0.95 block
        q_lt_95_blocks = [b for b in blocks if "(qs < 0.95)" in b.group()]
        assert len(q_lt_95_blocks) >= 1, (
            "Expected at least one parallel mask block with (qs < 0.95)"
        )

        for block in q_lt_95_blocks:
            text = block.group()
            assert "(coll == False)" in text, (
                "NC-04 regression: q < 0.95 mask must use (coll == False), "
                f"got:\n{textwrap.indent(text, '  ')}"
            )
            assert "(coll_twin == False)" not in text, (
                "NC-04 regression: q < 0.95 mask must NOT use "
                f"(coll_twin == False), got:\n{textwrap.indent(text, '  ')}"
            )

    def test_parallel_q_ge_95_still_uses_coll_twin(self):
        """The parallel q >= 0.95 mask must use (coll_twin == False) only."""
        source = self._read_lnZ_BEB_source()
        blocks = self._extract_parallel_mask_blocks(source)

        q_ge_95_blocks = [b for b in blocks if "(qs >= 0.95)" in b.group()]
        assert len(q_ge_95_blocks) >= 1, (
            "Expected at least one parallel mask block with (qs >= 0.95)"
        )

        for block in q_ge_95_blocks:
            text = block.group()
            assert "(coll_twin == False)" in text, (
                "q >= 0.95 mask must use (coll_twin == False), "
                f"got:\n{textwrap.indent(text, '  ')}"
            )
            # Every coll reference must be coll_twin, not plain coll
            coll_refs = re.findall(r"\bcoll(?:_twin)?\s*==\s*False", text)
            for ref in coll_refs:
                assert "coll_twin" in ref, (
                    "q >= 0.95 mask must NOT use plain (coll == False), "
                    f"got:\n{textwrap.indent(text, '  ')}"
                )


# ---------------------------------------------------------------------------
# NEB_unknown source-level regression test
# ---------------------------------------------------------------------------

class TestNEBUnknownCollisionMask:
    """NC-04: lnZ_NEB_unknown twin branch must use coll_twin, not coll.

    The q >= 0.95 parallel path in lnZ_NEB_unknown() used coll (standard-
    period collision mask) instead of coll_twin (twin-period collision mask).
    The serial code path at line 2849 already correctly used coll_twin[i].
    This is the inverse of the lnZ_BEB fix: there the standard branch used
    the twin mask; here the twin branch used the standard mask.
    """

    @staticmethod
    def _read_lnZ_NEB_unknown_source():
        """Return the full source of lnZ_NEB_unknown as a string."""
        from triceratops.marginal_likelihoods import lnZ_NEB_unknown
        return inspect.getsource(lnZ_NEB_unknown)

    @staticmethod
    def _extract_parallel_mask_blocks(source):
        """Extract multi-line mask blocks from the parallel path."""
        blocks = list(re.finditer(
            r"mask\s*=\s*\(.*?\n\s*\)",
            source,
            re.DOTALL,
        ))
        return [b for b in blocks if "qs" in b.group()]

    def test_parallel_q_ge_95_uses_coll_twin_not_coll(self):
        """The parallel q >= 0.95 mask must use (coll_twin == False)."""
        source = self._read_lnZ_NEB_unknown_source()
        blocks = self._extract_parallel_mask_blocks(source)

        q_ge_95_blocks = [b for b in blocks if "(qs >= 0.95)" in b.group()]
        assert len(q_ge_95_blocks) >= 1, (
            "Expected at least one parallel mask block with (qs >= 0.95)"
        )

        for block in q_ge_95_blocks:
            text = block.group()
            assert "(coll_twin == False)" in text, (
                "NC-04 regression: lnZ_NEB_unknown q >= 0.95 mask must use "
                f"(coll_twin == False), got:\n{textwrap.indent(text, '  ')}"
            )
            # Ensure it does NOT use plain (coll == False)
            # We need to check for exactly "coll ==" not preceded by "coll_twin"
            coll_refs = re.findall(r"coll(?:_twin)?\s*==\s*False", text)
            for ref in coll_refs:
                assert "coll_twin" in ref, (
                    "NC-04 regression: lnZ_NEB_unknown q >= 0.95 mask must "
                    "not use plain (coll == False), "
                    f"got:\n{textwrap.indent(text, '  ')}"
                )

    def test_parallel_q_lt_95_still_uses_coll(self):
        """The parallel q < 0.95 mask must still use (coll == False)."""
        source = self._read_lnZ_NEB_unknown_source()
        blocks = self._extract_parallel_mask_blocks(source)

        q_lt_95_blocks = [b for b in blocks if "(qs < 0.95)" in b.group()]
        assert len(q_lt_95_blocks) >= 1, (
            "Expected at least one parallel mask block with (qs < 0.95)"
        )

        for block in q_lt_95_blocks:
            text = block.group()
            assert "(coll == False)" in text, (
                "lnZ_NEB_unknown q < 0.95 mask must use (coll == False), "
                f"got:\n{textwrap.indent(text, '  ')}"
            )
            assert "(coll_twin == False)" not in text, (
                "lnZ_NEB_unknown q < 0.95 mask must NOT use "
                f"(coll_twin == False), got:\n{textwrap.indent(text, '  ')}"
            )


# ---------------------------------------------------------------------------
# Comprehensive check across ALL EB functions
# ---------------------------------------------------------------------------

class TestAllEBTwinBranches:
    """Verify every EB function's q >= 0.95 parallel mask uses coll_twin.

    This catches copy-paste errors across all EB functions, not just lnZ_BEB
    and lnZ_NEB_unknown. The pattern: every EB function has a parallel mask
    block for the twin period (q >= 0.95) that must use coll_twin, and a
    standard-period block (q < 0.95) that must use coll.
    """

    # All EB marginal likelihood functions that have twin branches
    EB_FUNCTIONS = [
        "lnZ_TEB",
        "lnZ_PEB",
        "lnZ_SEB",
        "lnZ_DEB",
        "lnZ_BEB",
        "lnZ_NEB_unknown",
        "lnZ_NEB_evolved",
    ]

    @staticmethod
    def _get_function_source(func_name):
        """Import and return the source of the named function."""
        import triceratops.marginal_likelihoods as ml
        func = getattr(ml, func_name)
        return inspect.getsource(func)

    @staticmethod
    def _find_parallel_twin_mask_blocks(source):
        """Find mask blocks in the parallel path that filter for q >= 0.95.

        Handles both multi-line and single-line mask assignment patterns.
        Returns list of (match_text, line_number) tuples.
        """
        results = []

        # Multi-line pattern: mask = (\n ...\n  )
        for m in re.finditer(
            r"mask\s*=\s*\(.*?\n\s*\)", source, re.DOTALL
        ):
            if "(qs >= 0.95)" in m.group() or "qs >= 0.95" in m.group():
                lineno = source[:m.start()].count("\n") + 1
                results.append((m.group(), lineno))

        # Single-line pattern: mask = (incs >= inc_min) & ... & (qs >= 0.95)
        for m in re.finditer(
            r"mask\s*=\s*\([^=\n]*qs\s*>=\s*0\.95[^)]*\)", source
        ):
            text = m.group()
            if text not in [r[0] for r in results]:
                lineno = source[:m.start()].count("\n") + 1
                results.append((text, lineno))

        return results

    @staticmethod
    def _find_parallel_standard_mask_blocks(source):
        """Find mask blocks in the parallel path that filter for q < 0.95."""
        results = []

        for m in re.finditer(
            r"mask\s*=\s*\(.*?\n\s*\)", source, re.DOTALL
        ):
            if "(qs < 0.95)" in m.group() or "qs < 0.95" in m.group():
                lineno = source[:m.start()].count("\n") + 1
                results.append((m.group(), lineno))

        for m in re.finditer(
            r"mask\s*=\s*\([^=\n]*qs\s*<\s*0\.95[^)]*\)", source
        ):
            text = m.group()
            if text not in [r[0] for r in results]:
                lineno = source[:m.start()].count("\n") + 1
                results.append((text, lineno))

        return results

    @pytest.mark.parametrize("func_name", EB_FUNCTIONS)
    def test_twin_branch_uses_coll_twin(self, func_name):
        """q >= 0.95 parallel mask in {func_name} must use coll_twin only."""
        source = self._get_function_source(func_name)
        blocks = self._find_parallel_twin_mask_blocks(source)

        assert len(blocks) >= 1, (
            f"{func_name}: expected at least one parallel mask block with "
            "q >= 0.95 (twin branch)"
        )

        for text, lineno in blocks:
            # Must contain coll_twin
            assert "coll_twin" in text, (
                f"{func_name} (approx line {lineno}): q >= 0.95 parallel "
                f"mask must reference coll_twin, got:\n{textwrap.indent(text, '  ')}"
            )
            # Every "coll ... == False" must be "coll_twin == False"
            coll_refs = re.findall(r"\bcoll(?:_twin)?\s*==\s*False", text)
            for ref in coll_refs:
                assert "coll_twin" in ref, (
                    f"{func_name} (approx line {lineno}): q >= 0.95 parallel "
                    f"mask uses plain 'coll == False' instead of "
                    f"'coll_twin == False', got:\n{textwrap.indent(text, '  ')}"
                )

    @pytest.mark.parametrize("func_name", EB_FUNCTIONS)
    def test_standard_branch_uses_coll(self, func_name):
        """q < 0.95 parallel mask in {func_name} must use coll only."""
        source = self._get_function_source(func_name)
        blocks = self._find_parallel_standard_mask_blocks(source)

        assert len(blocks) >= 1, (
            f"{func_name}: expected at least one parallel mask block with "
            "q < 0.95 (standard branch)"
        )

        for text, lineno in blocks:
            # Must contain coll == False
            assert re.search(r"\bcoll\s*==\s*False", text), (
                f"{func_name} (approx line {lineno}): q < 0.95 parallel "
                f"mask must reference coll, got:\n{textwrap.indent(text, '  ')}"
            )
            # Must NOT contain coll_twin == False
            assert "coll_twin" not in text, (
                f"{func_name} (approx line {lineno}): q < 0.95 parallel "
                f"mask must NOT use 'coll_twin == False', "
                f"got:\n{textwrap.indent(text, '  ')}"
            )
