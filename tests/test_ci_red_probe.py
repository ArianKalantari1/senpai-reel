def test_ci_must_go_red_on_failure():
    """Temporary probe: proves the CI check fails the build. Reverted immediately."""
    assert 1 == 2, "intentional failure to prove CI turns red"
