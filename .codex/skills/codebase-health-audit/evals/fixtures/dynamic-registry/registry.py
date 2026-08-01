PLUGINS = {}


def register(name):
    def decorator(func):
        PLUGINS[name] = func
        return func
    return decorator
