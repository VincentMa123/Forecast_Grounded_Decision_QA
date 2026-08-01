# Stale Fallback Fixture

`APP_TIMEOUT_SECONDS` and `DEFAULT_TIMEOUT_SECONDS` are the only supported
configuration contract. Invalid integer input must fail explicitly; do not
silently fall back to a default value.
