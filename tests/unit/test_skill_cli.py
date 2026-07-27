"""Unit tests for the aegis-deidentify CLI (monai_aegis/skill_cli.py).

Pins the wrapper-facing contract: stdout carries exactly one JSON envelope
(never logs), and exit codes map to run outcomes — 0 success, 1 error,
2 invalid input, 3 partial, 4 success-needs-review. The facade is stubbed;
its behavior is covered by test_api.py.
"""

import json

import pytest


from monai_aegis import api, envelope, skill_cli


def _env(status="success", needs_review=False):
    return {
        "envelope_version": envelope.ENVELOPE_VERSION,
        "tool": envelope.TOOL_NAME,
        "status": status,
        "input": "/in",
        "output_dir": "/out",
        "needs_manual_review": needs_review,
    }


def _run(monkeypatch, capsys, result=None, raises=None, argv=None):
    def fake_deidentify(**kwargs):
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(api, "deidentify", fake_deidentify)
    code = skill_cli.main(argv or ["/in", "--output-dir", "/out"])
    out = capsys.readouterr().out
    return code, json.loads(out), out


def test_success_exit_zero(monkeypatch, capsys):
    code, env, _ = _run(monkeypatch, capsys, result=_env())
    assert code == skill_cli.EXIT_SUCCESS
    assert env["status"] == "success"


def test_needs_review_exit_four(monkeypatch, capsys):
    code, _, _ = _run(monkeypatch, capsys, result=_env(needs_review=True))
    assert code == skill_cli.EXIT_NEEDS_REVIEW


def test_partial_exit_three(monkeypatch, capsys):
    code, _, _ = _run(monkeypatch, capsys, result=_env(status="partial"))
    assert code == skill_cli.EXIT_PARTIAL


def test_error_status_exit_one(monkeypatch, capsys):
    code, _, _ = _run(monkeypatch, capsys, result=_env(status="error"))
    assert code == skill_cli.EXIT_ERROR


def test_input_error_exit_two_with_envelope(monkeypatch, capsys):
    code, env, _ = _run(monkeypatch, capsys, raises=api.InputError("no such file: /in"))
    assert code == skill_cli.EXIT_INVALID_INPUT
    assert env["status"] == "error"
    assert "no such file" in env["message"]


def test_unexpected_error_exit_one_with_envelope(monkeypatch, capsys):
    code, env, _ = _run(monkeypatch, capsys, raises=RuntimeError("model load failed on patient X"))
    assert code == skill_cli.EXIT_ERROR
    assert env["status"] == "error"
    # PHI-safe: the envelope message is the exception type, not the raw text.
    assert env["message"] == "RuntimeError"
    assert "patient X" not in json.dumps(env)


def test_stdout_is_single_json_document(monkeypatch, capsys):
    _code, _parsed, out = _run(monkeypatch, capsys, result=_env())
    assert out.endswith("\n")
    json.loads(out)  # the whole stream parses as one document
    assert len(out.strip().splitlines()) == 1


def test_pretty_still_single_document(monkeypatch, capsys):
    code, env, out = _run(
        monkeypatch,
        capsys,
        result=_env(),
        argv=["/in", "--output-dir", "/out", "--pretty"],
    )
    assert code == skill_cli.EXIT_SUCCESS
    assert json.loads(out) == env


def test_review_priority_below_partial(monkeypatch, capsys):
    # A partial run that also flags review exits 3, not 4 — failures outrank
    # the review flag.
    code, _, _ = _run(monkeypatch, capsys, result=_env(status="partial", needs_review=True))
    assert code == skill_cli.EXIT_PARTIAL


def test_flags_are_passed_through(monkeypatch, capsys):
    seen = {}

    def fake_deidentify(**kwargs):
        seen.update(kwargs)
        return _env()

    monkeypatch.setattr(api, "deidentify", fake_deidentify)
    code = skill_cli.main(
        [
            "/in",
            "--output-dir",
            "/out",
            "--config",
            "/cfg/base.yaml",
            "--overlay",
            "/cfg/site.yaml",
            "--mode",
            "image",
        ]
    )
    assert code == skill_cli.EXIT_SUCCESS
    assert seen == {
        "input_path": "/in",
        "output_dir": "/out",
        "config_path": "/cfg/base.yaml",
        "overlay_path": "/cfg/site.yaml",
        "mode": "image",
    }
