class Cache_42:
    def __init__(self):
        self._store = {}
    def get(self, k):
        return self._store.get(k)
