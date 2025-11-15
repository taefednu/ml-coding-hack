import pandas as pd

from src import data_loading


def test_inventory_detects_target(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("client_id;issue_date;default_flag\n1;01.01.2023;0\n2;05.02.2023;1\n", encoding="cp1251")
    summary = data_loading.inventory_dataset(path)
    assert "default_flag" in summary.target_candidates
    assert summary.encoding.lower() in {"cp1251", "windows-1251", "ascii"}


def test_scan_data_dir_handles_multiple_formats(tmp_path):
    csv_path = tmp_path / "a.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
    json_path = tmp_path / "b.jsonl"
    json_path.write_text('{"x": 1, "y": 0}\n', encoding="utf-8")
    summaries = data_loading.scan_data_dir(tmp_path)
    assert len(summaries) == 2
