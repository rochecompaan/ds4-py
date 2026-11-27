import json
from pathlib import Path

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures/context_paging"


def test_manifest_covers_every_checked_in_fixture_exactly_once() -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
    fixture_paths = sorted(FIXTURE_ROOT.glob("*.json"))
    assert sorted(manifest["files"]) == [path.name for path in fixture_paths if path.name != "manifest.json"]

    fixture_ids: list[str] = []
    for filename in manifest["files"]:
        document = json.loads((FIXTURE_ROOT / filename).read_text())
        assert document["schemaVersion"] == manifest["schemaVersion"]
        for fixture in document["fixtures"]:
            assert {"id", "input", "expected"} <= set(fixture)
            fixture_ids.append(fixture["id"])

    covered_ids = [fixture_id for contract in manifest["coverage"] for fixture_id in contract["fixtures"]]
    assert len(fixture_ids) == len(set(fixture_ids))
    assert len(covered_ids) == len(set(covered_ids))
    assert set(covered_ids) == set(fixture_ids)
