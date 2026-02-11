"""Tests for custom error types."""

from __future__ import annotations

from featurization_utils.errors import ProcessorTypeError


def test_processor_type_error_message() -> None:
    err = ProcessorTypeError()
    assert "Processor type not recognized" in str(err)
