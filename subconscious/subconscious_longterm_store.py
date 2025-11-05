"""Long-term subconscious store (placeholder)."""

_store = {}

def store_longterm(key, value):
    _store[key] = value
    return True

def retrieve_longterm(key, default=None):
    return _store.get(key, default)
