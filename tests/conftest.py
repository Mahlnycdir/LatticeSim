"""
pytest configuration: hypothesis profiles for the property-based stress tests.

    pytest                                   # "default": deterministic, moderate number of examples
    HYPOTHESIS_PROFILE=thorough pytest -m "slow or not slow"   # many more examples
"""
import os

from hypothesis import HealthCheck, settings

settings.register_profile("default", max_examples=25, deadline=None, derandomize=True,
                          suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("thorough", max_examples=300, deadline=None, derandomize=False,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
