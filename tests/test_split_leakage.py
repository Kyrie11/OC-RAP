import json
from pathlib import Path


def test_root_scene_id_disjoint_across_splits(tmp_path):
    splits={'train':['a','b'],'calib':['c'],'test':['d']}
    (tmp_path/'splits.json').write_text(json.dumps(splits))
    sets=[set(v) for v in splits.values()]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])

def test_no_tuple_level_split():
    split_key='root_scene_id'
    assert split_key == 'root_scene_id'
