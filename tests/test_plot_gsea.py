"""Tests for gseapyvis.plot_gsea module."""

import os

import numpy as np
import pandas as pd
import pytest

from gseapyvis.plot_gsea import compute_heat_blocks, gsea_plot


# ---------------------------------------------------------------------------
# Helpers: synthetic data mimicking gseapy prerank result structure
# ---------------------------------------------------------------------------

TERM = "FAKE_TERM"


def _make_res_data(n_genes=100, n_hits=5, nes=1.5, pval=0.01, fdr=0.05):
    """Build a minimal res_data dict matching ``pre_res.results[term]``."""
    np.random.seed(42)
    res = np.cumsum(np.random.randn(n_genes))
    hits = sorted(np.random.choice(n_genes, size=n_hits, replace=False).tolist())
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    matched = ";".join(gene_names[h] for h in hits)
    return {
        "RES": res.tolist(),
        "hits": hits,
        "nes": nes,
        "pval": pval,
        "fdr": fdr,
        "matched_genes": matched,
    }


# Module-level class so joblib can pickle it
class FakePreRes:
    def __init__(self, res_data=None):
        if res_data is None:
            res_data = _make_res_data()
        self.res2d = pd.DataFrame({"Term": [TERM]})
        self.results = {TERM: res_data}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def res_data():
    return _make_res_data()


@pytest.fixture
def res_data_neg_nes():
    return _make_res_data(nes=-1.2)


@pytest.fixture
def rnk_df():
    return pd.DataFrame({
        "gene": [f"Gene{i}" for i in range(100)],
        "score": np.random.randn(100).tolist(),
    })


@pytest.fixture
def gsdata():
    """Minimal gsdata for compute_heat_blocks with 20 positions, 3 hits."""
    df = pd.DataFrame({"position": [0] * 20})
    df["Description"] = TERM
    df.loc[[2, 7, 15], "position"] = 1
    return df


@pytest.fixture
def pkl_path(tmp_path):
    """Create a temporary .pkl with FakePreRes and return its path."""
    import joblib
    path = tmp_path / "test.pkl"
    joblib.dump(FakePreRes(), path)
    return path


# ---------------------------------------------------------------------------
# compute_heat_blocks tests
# ---------------------------------------------------------------------------

class TestComputeHeatBlocks:
    def test_returns_dataframe(self, gsdata):
        result = compute_heat_blocks(gsdata)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, gsdata):
        result = compute_heat_blocks(gsdata)
        for col in ("xmin", "xmax", "ymin", "ymax", "col", "Description"):
            assert col in result.columns

    def test_blocks_cover_full_range(self, gsdata):
        result = compute_heat_blocks(gsdata)
        assert result["xmin"].min() == 0
        assert result["xmax"].max() == len(gsdata)

    def test_ymin_zero_ymax_default(self, gsdata):
        result = compute_heat_blocks(gsdata)
        assert (result["ymin"] == 0).all()
        assert (result["ymax"] == 1.0).all()

    def test_custom_height(self, gsdata):
        result = compute_heat_blocks(gsdata, htHeight=2.5)
        assert (result["ymax"] == 2.5).all()

    def test_col_values_are_hex_colors(self, gsdata):
        result = compute_heat_blocks(gsdata)
        for c in result["col"]:
            assert c.startswith("#"), f"Expected hex color, got {c}"
            assert len(c) == 7, f"Expected 7-char hex, got {c}"

    def test_multiple_setids(self):
        df = pd.DataFrame({"position": [0] * 10})
        df["Description"] = "SET_A"
        df.loc[[1, 5], "position"] = 1
        df2 = pd.DataFrame({"position": [0] * 10})
        df2["Description"] = "SET_B"
        df2.loc[[3, 7], "position"] = 1
        combined = pd.concat([df, df2], ignore_index=True)

        result = compute_heat_blocks(combined)
        assert set(result["Description"]) == {"SET_A", "SET_B"}
        assert len(result) > 0

    def test_no_hits(self):
        df = pd.DataFrame({"position": [0] * 20})
        df["Description"] = "EMPTY"
        result = compute_heat_blocks(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# gsea_plot tests
# ---------------------------------------------------------------------------

class TestGseaPlot:
    def test_returns_plot_object(self, res_data):
        p = gsea_plot({TERM: res_data}, TERM)
        assert p is not None

    def test_plot_with_rnk(self, res_data, rnk_df):
        p = gsea_plot({TERM: res_data}, TERM, rnk=rnk_df)
        assert p is not None

    def test_negative_nes(self, res_data_neg_nes):
        p = gsea_plot({TERM: res_data_neg_nes}, TERM)
        assert p is not None

    def test_without_rnk(self, res_data):
        p = gsea_plot({TERM: res_data}, TERM, rnk=None)
        assert p is not None

    def test_rnk_merge_adds_score(self, res_data, rnk_df):
        p = gsea_plot({TERM: res_data}, TERM, rnk=rnk_df)
        assert p is not None

    def test_invalid_term_raises(self, res_data):
        with pytest.raises(KeyError):
            gsea_plot({TERM: res_data}, "NONEXISTENT_TERM")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_missing_file_exits(self):
        from typer.testing import CliRunner
        from gseapyvis.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["plot", "nonexistent.pkl"])
        assert result.exit_code != 0

    def test_invalid_extension_exits(self, pkl_path):
        from typer.testing import CliRunner
        from gseapyvis.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["plot", str(pkl_path), "-o", "out.txt"])
        assert result.exit_code != 0

    def test_svg_export(self, pkl_path, tmp_path):
        from typer.testing import CliRunner
        from gseapyvis.cli import app

        out_path = tmp_path / "gsea.svg"
        runner = CliRunner()
        result = runner.invoke(app, ["plot", str(pkl_path), "-o", str(out_path)])
        assert result.exit_code == 0
        assert out_path.exists()
        assert "<svg" in out_path.read_text().lower()

    def test_html_export(self, pkl_path, tmp_path):
        from typer.testing import CliRunner
        from gseapyvis.cli import app

        out_path = tmp_path / "gsea.html"
        runner = CliRunner()
        result = runner.invoke(app, ["plot", str(pkl_path), "-o", str(out_path)])
        assert result.exit_code == 0
        assert out_path.exists()
        assert "<html" in out_path.read_text().lower() or "lets-plot" in out_path.read_text()

    def test_to_buffer(self, pkl_path):
        from typer.testing import CliRunner
        from gseapyvis.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["plot", str(pkl_path), "--to-buffer"])
        assert result.exit_code == 0
