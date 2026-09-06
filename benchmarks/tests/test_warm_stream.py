"""Le chemin destructeur de ``warm_stream`` : un fichier partagé ne doit jamais partir.

La première version de la garde lisait le nom de fichier. Sur les 287 ``.mat`` réels de
stieger2021 elle produisait 570 revendications pour 287 fichiers -- le sujet 1 réclamait
``S2_Session_1.mat``, parce que le numéro de *session* matche aussi le motif. Lancée
telle quelle, elle aurait effacé les fichiers de 61 autres sujets. Ces tests fixent le
comportement qui a remplacé l'heuristique.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


class _FakeDataset:
    def __init__(self, layout: dict, code: str = "fake"):
        self.code = code
        self._layout = layout
        self.subject_list = sorted(layout)

    def data_path(self, subject, **_):
        return [str(p) for p in self._layout[subject]]


def _run(monkeypatch, tmp_path, layout, extra_argv=()):
    """Exécuter ``warm_stream.main`` sur une topologie de fichiers donnée."""
    import warm_stream

    ds = _FakeDataset(layout)
    par = types.SimpleNamespace(get_data=lambda **kw: None)

    monkeypatch.setattr(warm_stream, "build_campaign", lambda name: (ds, par, {"ab": "epochs"}))
    monkeypatch.setattr(warm_stream, "subject_complete", lambda *a, **k: True)
    monkeypatch.setattr(warm_stream, "quota_free_gb", lambda: 10_000)

    # Les imports tardifs de main() : on les neutralise, rien ici ne touche MOABB.
    fake_bids = types.ModuleType("moabb.datasets.bids_interface")
    fake_bids.get_bids_root = lambda code, _: Path(tmp_path / "cache" / f"MNE-BIDS-{code}")
    monkeypatch.setitem(sys.modules, "moabb.datasets.bids_interface", fake_bids)
    fake_utils = types.ModuleType("utils")
    fake_utils.cache_config = lambda cfg: None
    monkeypatch.setitem(sys.modules, "utils", fake_utils)

    monkeypatch.setattr(sys, "argv",
                        ["warm_stream.py", "--dataset", "fake",
                         "--cache", str(tmp_path / "cache"), *extra_argv])
    return warm_stream.main()


def _make(tmp_path: Path, names) -> dict:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for p in names:
        f = raw / p
        f.write_bytes(b"x" * 1024)
        out[p] = f
    return out


def test_fichiers_par_sujet_tout_efface(monkeypatch, tmp_path):
    """Topologie stieger2021 : chaque sujet a ses fichiers, tout doit finir effacé."""
    files = _make(tmp_path, [f"S{s}_Session_{e}.mat" for s in (1, 2, 17) for e in (1, 2)])
    layout = {s: [files[f"S{s}_Session_{e}.mat"] for e in (1, 2)] for s in (1, 2, 17)}

    assert _run(monkeypatch, tmp_path, layout) == 0
    assert not [f for f in files.values() if f.exists()]


def test_archive_partagee_rien_efface(monkeypatch, tmp_path):
    """Topologie yang2025 : un seul fichier pour tous les sujets, il doit survivre."""
    files = _make(tmp_path, ["tout.zip"])
    layout = {s: [files["tout.zip"]] for s in (1, 2, 3)}

    assert _run(monkeypatch, tmp_path, layout) == 0
    assert files["tout.zip"].exists(), "une archive partagée a été effacée"


def test_fichier_partage_par_deux_sujets_survit(monkeypatch, tmp_path):
    """Le cas mixte : le commun reste, le propre part."""
    files = _make(tmp_path, ["commun.mat", "propre_1.mat", "propre_2.mat"])
    layout = {
        1: [files["commun.mat"], files["propre_1.mat"]],
        2: [files["commun.mat"], files["propre_2.mat"]],
    }

    assert _run(monkeypatch, tmp_path, layout) == 0
    assert files["commun.mat"].exists(), "le fichier réclamé par 2 sujets a été effacé"
    assert not files["propre_1.mat"].exists()
    assert not files["propre_2.mat"].exists()


def test_le_sujet_suivant_protege_le_precedent(monkeypatch, tmp_path):
    """Le retard d'un sujet : un fichier réclamé PLUS TARD ne doit pas déjà être parti.

    C'est la raison d'être du retard. Sans lui, le fichier du sujet 1 serait effacé avant
    qu'on découvre que le sujet 2 le réclame aussi.
    """
    files = _make(tmp_path, ["a.mat", "b.mat"])
    layout = {1: [files["a.mat"]], 2: [files["a.mat"], files["b.mat"]]}

    assert _run(monkeypatch, tmp_path, layout) == 0
    assert files["a.mat"].exists(), "effacé avant que le sujet suivant le réclame"
    assert not files["b.mat"].exists()


def test_dry_run_n_efface_rien(monkeypatch, tmp_path):
    files = _make(tmp_path, ["S1_Session_1.mat", "S2_Session_1.mat"])
    layout = {1: [files["S1_Session_1.mat"]], 2: [files["S2_Session_1.mat"]]}

    assert _run(monkeypatch, tmp_path, layout, extra_argv=("--dry-run",)) == 0
    assert all(f.exists() for f in files.values())


def test_plancher_de_quota_refuse(monkeypatch, tmp_path):
    import warm_stream
    monkeypatch.setattr(warm_stream, "quota_free_gb", lambda: 3)
    files = _make(tmp_path, ["S1.mat"])
    layout = {1: [files["S1.mat"]]}

    monkeypatch.setattr(warm_stream, "build_campaign",
                        lambda name: pytest.fail("ne doit pas aller jusque-là"))
    monkeypatch.setattr(sys, "argv",
                        ["warm_stream.py", "--dataset", "fake",
                         "--cache", str(tmp_path / "cache"), "--floor-gb", "25"])
    assert warm_stream.main() == 3
    assert files["S1.mat"].exists()
