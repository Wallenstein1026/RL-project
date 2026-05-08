"""Tests for the canonical experiment entry point defaults."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main import parse_args


def test_default_num_runs_matches_final_report_setting(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = parse_args()

    assert args.num_runs == 8
