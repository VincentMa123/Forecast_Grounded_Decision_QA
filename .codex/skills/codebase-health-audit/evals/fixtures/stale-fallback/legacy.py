LEGACY_TIMEOUT_SECONDS = 45


def legacy_timeout(env):
    try:
        return int(env["OLD_TIMEOUT"])
    except Exception:
        return LEGACY_TIMEOUT_SECONDS
