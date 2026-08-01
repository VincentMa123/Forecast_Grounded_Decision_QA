DEFAULT_TIMEOUT_SECONDS = 30


def timeout_seconds(env):
    return int(env.get("APP_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
