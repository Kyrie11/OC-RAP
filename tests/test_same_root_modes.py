from ocrap.teacher.root_modes import generate_root_modes, build_latent_context, MODE_SLOT_SEMANTICS


def test_same_root_mode_schedule_identical_across_actions():
    modes=generate_root_modes(7, M=4)
    c1=build_latent_context('root', modes[2])
    c2=build_latent_context('root', modes[2])
    assert c1 == c2
    assert c1.deterministic_subseed('root','traffic') == c2.deterministic_subseed('root','traffic')

def test_mode_seed_not_resampled_per_option():
    modes=generate_root_modes(9, M=4)
    seeds=[m.rng_seed for m in modes]
    assert seeds == [m.rng_seed for m in generate_root_modes(9, M=4)]

def test_root_shared_mode_is_latent_context_not_open_loop_trajectory():
    ctx=build_latent_context('root', generate_root_modes(1, M=2)[1])
    assert hasattr(ctx,'aggressiveness') and not hasattr(ctx,'open_loop_trajectory')

def test_care_mode_query_fixed_semantic_index_alignment():
    modes=generate_root_modes(0, M=8)
    assert [m.semantic for m in modes] == MODE_SLOT_SEMANTICS[:8]
