import plugin
from registry import PLUGINS


def run(name, value):
    return PLUGINS[name](value)
