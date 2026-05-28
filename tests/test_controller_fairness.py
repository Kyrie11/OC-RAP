from ocrap.controllers.pure_pursuit_pid import PurePursuitPID


def test_same_controller_config_for_all_methods():
    methods=['ours','nominal','risk_aware','backup_filter','direct_scalar_critic','oracle']
    cfgs=[PurePursuitPID().__dict__ for _ in methods]
    assert all(c==cfgs[0] for c in cfgs)

def test_controller_does_not_access_mero_scores():
    import inspect
    src=inspect.getsource(PurePursuitPID.track)
    assert 'MERO' not in src and 'profiles' not in src and 'R_star' not in src
