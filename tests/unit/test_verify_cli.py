"""Unit tests for the aegis-verify CLI (monai_aegis/verify_cli.py).

Pins the wrapper contract: one JSON document on stdout and exit codes
0 pass / 1 engine error / 2 invalid input / 3 checks failed. The engine is
stubbed; its behavior is covered by test_verify.py.
"""

import json


from monai_aegis import verify, verify_cli
from monai_aegis.api import InputError


def _report(status="pass", failures=0):
    return {
        "report_version": verify.REPORT_VERSION,
        "tool": verify.TOOL_NAME,
        "checklist": "ps315-deidentification",
        "run_dir": "/out",
        "status": status,
        "totals": {"files_checked": 1, "checks_evaluated": 10, "failures": failures, "warnings": 0},
        "findings": [],
    }


def _run(monkeypatch, capsys, result=None, raises=None, argv=None):
    def fake_verify_run(run_dir, checklist=None):
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(verify, "verify_run", fake_verify_run)
    code = verify_cli.main(argv or ["/out"])
    out = capsys.readouterr().out
    return code, json.loads(out), out


def test_pass_exit_zero(monkeypatch, capsys):
    code, report, _ = _run(monkeypatch, capsys, result=_report())
    assert code == verify_cli.EXIT_PASS
    assert report["status"] == "pass"


def test_fail_exit_three(monkeypatch, capsys):
    code, _, _ = _run(monkeypatch, capsys, result=_report(status="fail", failures=2))
    assert code == verify_cli.EXIT_FAIL


def test_invalid_input_exit_two(monkeypatch, capsys):
    code, doc, _ = _run(monkeypatch, capsys, raises=InputError("no such directory: /out"))
    assert code == verify_cli.EXIT_INVALID_INPUT
    assert doc["status"] == "error"


def test_engine_error_exit_one(monkeypatch, capsys):
    code, doc, _ = _run(monkeypatch, capsys, raises=ValueError("unknown check type 'x'"))
    assert code == verify_cli.EXIT_ERROR
    assert "ValueError" in doc["message"]


def test_stdout_single_json_document(monkeypatch, capsys):
    _code, _doc, out = _run(monkeypatch, capsys, result=_report())
    assert len(out.strip().splitlines()) == 1
    json.loads(out)


def test_checklist_flag_passed_through(monkeypatch, capsys):
    seen = {}

    def fake_verify_run(run_dir, checklist=None):
        seen.update(run_dir=run_dir, checklist=checklist)
        return _report()

    monkeypatch.setattr(verify, "verify_run", fake_verify_run)
    code = verify_cli.main(["/out", "--checklist", "/cfg/site.yaml"])
    assert code == verify_cli.EXIT_PASS
    assert seen == {"run_dir": "/out", "checklist": "/cfg/site.yaml"}
