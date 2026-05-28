from ocrap.envs.metadrive_adapter import MetaDriveStateAdapter


class Obj:
    def __init__(self, oid, x, y, speed=0.0):
        self.id = oid
        self.position = (x, y)
        self.heading_theta = 0.0
        self.speed = speed


class Engine:
    def __init__(self, agent, other):
        self.agent = agent
        self._objects = [agent, other]

    def get_objects(self):
        return list(self._objects)


class Env:
    def __init__(self):
        self.agent = Obj("default_agent", 0.0, 0.0)
        self.engine = Engine(self.agent, Obj("traffic_0", 10.0, 0.0, speed=2.0))


def test_metadrive_actor_adapter_filters_ego_from_engine_objects():
    actors = MetaDriveStateAdapter(strict=False).get_actor_states(Env())
    ids = [a.actor_id for a in actors]
    assert "default_agent" not in ids
    assert ids == ["traffic_0"]
