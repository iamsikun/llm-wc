"""Criteria D1, D3, D4, D5: the site builds offline and passes its own checks."""
import json

import pytest

from llm_wc.tracker import registry
from llm_wc.tracker.site.build import build


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return build(tmp_path_factory.mktemp("site") / "_site")


def test_every_page_is_built(site):
    for page in ("index.html", "benchmarks.html", "methods.html", "design.html",
                 "about.html", "404.html", ".nojekyll", "og/home.png"):
        assert (site / page).is_file()
    for key, spec in registry.settings().items():
        if spec.get("in_reference"):
            assert (site / "benchmarks" / f"{key}.html").is_file()
            assert json.loads((site / "data" / key / "reference.json").read_text())["edition"] == "reference"
            assert (site / "og" / f"{key}.png").read_bytes()[:4] == b"\x89PNG"
            assert (site / "badges" / f"{key}.svg").read_text().startswith("<svg")


def test_no_external_scripts_or_styles(site):
    for page in site.rglob("*.html"):
        html = page.read_text()
        assert "<script src=\"http" not in html and "stylesheet\" href=\"http" not in html, page


def test_reference_edition_csvs_are_the_committed_results(site):
    from llm_wc.tracker.paths import REFERENCE_AUDIT

    for key, spec in registry.settings().items():
        if spec.get("in_reference"):
            assert (site / "data" / key / "reference-leaderboard.csv").read_bytes() == (REFERENCE_AUDIT / key / "models.csv").read_bytes()
