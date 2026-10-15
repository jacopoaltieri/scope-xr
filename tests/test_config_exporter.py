import sys

from importlib.resources import files

from scopexr.config_exporter import CONFIG_FILES, export_configs


def test_export_configs_writes_all_default_configs(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "configs"
    monkeypatch.setattr(
        sys,
        "argv",
        ["scopexr-confex", "--output-dir", str(output_dir)],
    )

    export_configs()

    assert output_dir.is_dir()
    output = capsys.readouterr().out
    for filename in CONFIG_FILES:
        exported = output_dir / filename
        bundled = files("scopexr").joinpath("default_configs", filename)
        assert exported.read_bytes() == bundled.read_bytes()
        assert f"Exported {filename} -> {exported.resolve()}" in output


def test_export_configs_uses_current_directory_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["scopexr-confex"])

    export_configs()

    assert {path.name for path in tmp_path.iterdir()} == set(CONFIG_FILES)
    assert all((tmp_path / filename).is_file() for filename in CONFIG_FILES)
