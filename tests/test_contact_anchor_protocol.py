from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from ocrap.evaluation.contact_anchor import simulator_state_fingerprint, select_one_anchor_per_scene


def _state(x0: float = 0.0):
    tr = SimpleNamespace(
        x=np.asarray([[x0, x0 + 1.0], [10.0, 11.0]]),
        y=np.asarray([[0.0, 0.0], [2.0, 2.0]]),
        z=np.zeros((2, 2)), yaw=np.zeros((2, 2)),
        vel_x=np.ones((2, 2)), vel_y=np.zeros((2, 2)),
        length=np.full((2, 2), 4.5), width=np.full((2, 2), 2.0), height=np.full((2, 2), 1.5),
        valid=np.ones((2, 2), dtype=bool),
    )
    meta = SimpleNamespace(ids=np.asarray([1, 2]), object_types=np.asarray([1, 1]), is_sdc=np.asarray([True, False]))
    return SimpleNamespace(timestep=np.asarray(0), sim_trajectory=tr, object_metadata=meta)


def test_simulator_state_fingerprint_is_stable_and_state_sensitive():
    a = simulator_state_fingerprint(_state(0.0))
    b = simulator_state_fingerprint(_state(0.0))
    c = simulator_state_fingerprint(_state(0.01))
    assert len(a) == 64 and a == b and a != c


def test_anchor_manifest_is_scene_disjoint_and_uses_only_pretreatment_ties():
    rows = [
        {"scene_id": "s1", "target_key": "s1:t10", "target_time_index": 10, "contact_anchor_found": True,
         "contact_anchor_remaining_steps": 15, "contact_anchor_time_index": 20, "contact_anchor_prelude_env_steps": 10,
         "contact_anchor_fingerprint": "a" * 64},
        {"scene_id": "s1", "target_key": "s1:t12", "target_time_index": 12, "contact_anchor_found": True,
         "contact_anchor_remaining_steps": 25, "contact_anchor_time_index": 19, "contact_anchor_prelude_env_steps": 7,
         "contact_anchor_fingerprint": "b" * 64},
        {"scene_id": "s2", "target_key": "s2:t9", "target_time_index": 9, "contact_anchor_found": True,
         "contact_anchor_remaining_steps": 5, "contact_anchor_time_index": 12, "contact_anchor_prelude_env_steps": 3,
         "contact_anchor_fingerprint": "c" * 64},
    ]
    m = select_one_anchor_per_scene({"scenes": rows}, min_post_steps=10)
    assert m["valid"] is True
    assert m["num_selected_anchors"] == m["num_selected_scenes"] == 1
    assert m["target_keys"] == ["s1:t12"]
    assert m["anchors"][0]["contact_anchor_fingerprint"] == "b" * 64
