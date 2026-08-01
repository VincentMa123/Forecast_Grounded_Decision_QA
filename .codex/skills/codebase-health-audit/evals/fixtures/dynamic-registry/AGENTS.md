# Dynamic Registry Fixture

Plugins are discovered by decorator-based registration. Imports may execute a
registration decorator, so a plugin function is reachable through `PLUGINS`
even when it has no direct call site.
