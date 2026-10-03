"""Checks that the package is installed and importable."""

import dlv


def test_package_has_docstring():
    """The installed package exposes its module docstring."""
    assert "Computer Vision" in dlv.__doc__
