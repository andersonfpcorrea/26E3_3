"""Checks that the package is installed and importable."""

import dlv


def test_package_has_docstring():
    assert "Computer Vision" in dlv.__doc__
