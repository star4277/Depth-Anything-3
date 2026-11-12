import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

dispatcher = importlib.import_module("depth_anything_3.utils.export")


def test_import_does_not_require_torch_or_optional_exporters():
    source = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'moviepy', 'pycolmap', 'gsplat', 'trimesh', 'matplotlib'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
from depth_anything_3.utils.export import export, SUPPORTED_EXPORT_FORMATS
assert callable(export) and 'mini_npz' in SUPPORTED_EXPORT_FORMATS
"""
    subprocess.run([sys.executable, "-c", source], check=True)


@pytest.mark.parametrize("name", ["", "unknown", "npz-unknown", "npz-", None])
def test_invalid_compound_format_has_no_side_effects(monkeypatch, tmp_path, name):
    def no_import(*args):
        pytest.fail("invalid format reached the importer")
    monkeypatch.setattr(dispatcher, "import_module", no_import)
    with pytest.raises(ValueError, match="Unsupported export format"):
        dispatcher.export(object(), name, str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_compound_dispatch_preserves_order_and_options(monkeypatch, tmp_path):
    calls = []
    def load(module, package):
        if module == ".npz":
            return SimpleNamespace(export_to_mini_npz=lambda *a, **kw: calls.append(("npz", kw)))
        return SimpleNamespace(export_to_glb=lambda *a, **kw: calls.append(("glb", kw)))
    monkeypatch.setattr(dispatcher, "import_module", load)
    dispatcher.export(object(), "mini_npz-glb", str(tmp_path), glb={"show_cameras": False})
    assert calls == [("npz", {}), ("glb", {"show_cameras": False})]


def test_missing_optional_dependency_names_the_extra(monkeypatch, tmp_path):
    def missing(*args):
        raise ModuleNotFoundError("No pycolmap", name="pycolmap")
    monkeypatch.setattr(dispatcher, "import_module", missing)
    with pytest.raises(ImportError, match=r"depth-anything-3\[colmap\]"):
        dispatcher.export(object(), "colmap", str(tmp_path))


def test_missing_internal_module_is_not_masked(monkeypatch, tmp_path):
    def broken(*args):
        raise ModuleNotFoundError("internal defect", name="depth_anything_3.broken")
    monkeypatch.setattr(dispatcher, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="internal defect"):
        dispatcher.export(object(), "glb", str(tmp_path))
