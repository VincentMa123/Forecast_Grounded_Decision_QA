from legacy import legacy_timeout
from settings import timeout_seconds


def resolved_timeout(env, use_legacy=False):
    if use_legacy:
        return legacy_timeout(env)
    return timeout_seconds(env)
