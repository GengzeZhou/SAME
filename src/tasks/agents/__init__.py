from .base_agent import MetaAgent

# import the agent class here
from .navillm_agent import NaviLLMAgent
from .duet_agent import DUETAgent


def load_agent(name, *args, **kwargs):
    cls = MetaAgent.registry[name]
    return cls(*args, **kwargs)