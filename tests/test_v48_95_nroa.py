from ocrap.v48_95_native_recovery_observability import cert4, frozen_native_features, tie_auc


def test_tie_auc():
    assert tie_auc([1,2],[0,0]) == 1.0
    assert tie_auc([0],[0]) == 0.5


def test_native_features_are_exact_deltas():
    x=frozen_native_features([.3,.7,.4,.9],[.1,.6,.2,1.0])
    assert abs(x['delta_hard_support']-.2)<1e-12
    assert abs(x['delta_deployability']-.1)<1e-12
    assert abs(x['delta_smooth_support']-.2)<1e-12
    assert abs(x['delta_gap_quality']+.1)<1e-12


def test_cert_rejects_bad_shape():
    try: cert4([1,2],name='x')
    except ValueError: pass
    else: raise AssertionError('expected ValueError')
