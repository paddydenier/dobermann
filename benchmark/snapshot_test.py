import pytest

from benchmark import __main__ as cli
from benchmark import snapshot
from benchmark.runner import DocumentResult


def doc(index, lengths, pk=0.5):
    return DocumentResult(
        index=index,
        reference_lengths=[2, 2],
        segment_lengths=lengths,
        metrics={"pk_default": pk, "wd_default": pk, "ghd": 1.0},
    )


def results(*documents, segmenter="text_tiling", dataset="CHOI"):
    return {segmenter: {dataset: list(documents)}}


def test_save_and_load_round_trip(tmp_path):
    original = results(doc(0, [2, 2]), doc(1, [4]))

    path = snapshot.save(original, "quick", tmp_path)
    loaded = snapshot.load("quick", tmp_path)

    assert path == tmp_path / "quick.json"
    assert loaded.results == original
    assert loaded.meta["mode"] == "quick"
    assert {"commit", "dirty", "created", "versions"} <= loaded.meta.keys()


def test_load_without_baseline(tmp_path):
    assert snapshot.load("quick", tmp_path) is None


def test_compare_identical_results():
    comparison = snapshot.compare(results(doc(0, [2, 2])), results(doc(0, [2, 2])))

    assert not comparison.has_changes
    assert [d.status for d in comparison.differences] == ["unchanged"]


def test_compare_detects_changed_documents():
    baseline = results(doc(0, [2, 2], pk=0.5), doc(1, [4], pk=0.5))
    current = results(doc(0, [2, 2], pk=0.5), doc(1, [1, 3], pk=0.25))

    (difference,) = snapshot.compare(baseline, current).differences

    assert difference.status == "changed"
    assert difference.changed == [1]
    assert difference.before["pk_default"] == 0.5
    assert difference.after["pk_default"] == 0.375


def test_compare_counts_documents_on_one_side_as_changed():
    baseline = results(doc(0, [4]))
    current = results(doc(0, [4]), doc(1, [4]))

    (difference,) = snapshot.compare(baseline, current).differences

    assert difference.changed == [1]
    assert difference.documents == 2


def test_compare_reports_new_and_missing_segmenters():
    baseline = results(doc(0, [4]), segmenter="old")
    current = results(doc(0, [4]), segmenter="new")

    comparison = snapshot.compare(baseline, current)

    assert comparison.has_changes
    statuses = {d.segmenter: d.status for d in comparison.differences}
    assert statuses == {"old": "missing", "new": "new"}


def test_format_report_lists_most_changed_documents():
    baseline = results(doc(0, [4], pk=0.5))
    current = results(doc(0, [1, 3], pk=0.25))

    report = snapshot.format_report(snapshot.compare(baseline, current))

    assert "text_tiling   CHOI" in report
    assert "1/1" in report
    assert "text_tiling / CHOI #0: pk_default 0.5000 -> 0.2500" in report
    assert "before [4]" in report
    assert "after  [1, 3]" in report


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Point the cli at a temporary snapshot folder and canned results."""
    current = {"results": results(doc(0, [2, 2]))}
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(cli, "_run", lambda full: current["results"])
    return current


def test_check_without_baseline(fake_cli, capsys):
    assert cli.main(["check"]) == 2
    assert "python -m benchmark save" in capsys.readouterr().err


def test_check_after_save_without_changes(fake_cli, capsys):
    assert cli.main(["save"]) == 0
    assert cli.main(["check"]) == 0
    assert "All segmentations match the baseline." in capsys.readouterr().out


def test_check_after_behaviour_change(fake_cli, capsys):
    assert cli.main(["save"]) == 0
    fake_cli["results"] = results(doc(0, [1, 3]))

    assert cli.main(["check"]) == 1
    assert "Segmentations differ from the baseline." in capsys.readouterr().out


def test_quick_and_full_baselines_are_separate(fake_cli, tmp_path):
    assert cli.main(["save"]) == 0

    assert (tmp_path / "quick.json").exists()
    assert cli.main(["check", "--full"]) == 2


def test_save_refuses_to_overwrite_baseline(fake_cli, tmp_path, capsys):
    assert cli.main(["save"]) == 0
    before = (tmp_path / "quick.json").read_bytes()
    capsys.readouterr()

    fake_cli["results"] = results(doc(0, [1, 3]))
    assert cli.main(["save"]) == 2

    assert (tmp_path / "quick.json").read_bytes() == before
    assert "--force" in capsys.readouterr().err


def test_save_with_force_replaces_baseline(fake_cli, tmp_path):
    assert cli.main(["save"]) == 0
    before = (tmp_path / "quick.json").read_bytes()

    fake_cli["results"] = results(doc(0, [1, 3]))
    assert cli.main(["save", "--force"]) == 0

    assert (tmp_path / "quick.json").read_bytes() != before
    assert cli.main(["check"]) == 0


@pytest.mark.parametrize("command", ["run", "check"])
def test_force_is_only_accepted_by_save(fake_cli, command):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--force"])
    assert error.value.code == 2
