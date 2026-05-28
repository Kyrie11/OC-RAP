import copy
from ocrap.envs.root_state import RootStateRestorer

class DummyEnv:
    def __init__(self):
        self.ego_state = {"x": 0.0, "v": 1.0}
        self.actor_states = [{"id": "a", "x": 5.0}]
        self.traffic_policy_states = {"aggr": 0.1}
        self.navigation_state = {"s": 0.0}
        self.tick = 0
        self.map_config = {"map": "S"}
        self.traffic_config = {"density": 0.1}
        self.history = [0]
    def step(self, action):
        self.tick += 1
        self.ego_state["x"] += action
        self.actor_states[0]["x"] += 0.5
        return self.ego_state

def rollout(env, actions):
    out=[]
    for a in actions:
        env.step(a); out.append((copy.deepcopy(env.ego_state), copy.deepcopy(env.actor_states), env.tick))
    return out

def test_root_restore_determinism():
    env=DummyEnv(); rest=RootStateRestorer(); root=rest.save(env)
    actions=[1.0,2.0,-0.5]
    rest.restore(env,root); a=rollout(env,actions)
    rest.restore(env,root); b=rollout(env,actions)
    assert a==b

def test_deterministic_replay_fallback():
    env=DummyEnv(); rest=RootStateRestorer(); root=rest.save(env); root.snapshot_supported=False
    rest.restore(env,root)
    assert env.tick==0 and env.actor_states[0]["x"]==5.0

def test_restore_includes_actor_and_policy_state():
    env=DummyEnv(); rest=RootStateRestorer(); root=rest.save(env)
    env.actor_states[0]["x"]=99; env.traffic_policy_states["aggr"]=1.0
    rest.restore(env,root)
    assert env.actor_states[0]["x"]==5.0
    assert env.traffic_policy_states["aggr"]==0.1
