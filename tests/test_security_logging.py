import logging
from litscout.cli import _RedactingFormatter, _is_probably_valid_key


def test_key_validation():
    assert _is_probably_valid_key("abcdef12345")
    assert not _is_probably_valid_key("short")
    assert not _is_probably_valid_key("has space")
    assert not _is_probably_valid_key("")
    assert not _is_probably_valid_key(None)


def test_log_redacts_sensitive_fields():
    formatter = _RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="request %s",
        args=({"api_key": "secret", "token": "abc", "normal": "ok"},),
        exc_info=None,
    )
    message = formatter.format(record)
    assert "secret" not in message
    assert "abc" not in message
    assert "'normal': 'ok'" in message


def test_log_redacts_bearer_token_in_message():
    formatter = _RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=20,
        msg="Authorization: Bearer secret-token",
        args=(),
        exc_info=None,
    )
    message = formatter.format(record)
    assert "secret-token" not in message
    assert "Bearer ***" in message
