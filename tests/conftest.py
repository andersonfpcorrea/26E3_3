"""Shared pytest setup: headless plotting and figure cleanup."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    """Closes every matplotlib figure after each test."""
    yield
    plt.close("all")
