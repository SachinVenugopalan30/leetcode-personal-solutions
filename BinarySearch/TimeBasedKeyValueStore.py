# 981. Time Based Key-Value Store


class TimeMap:
    def __init__(self):
        self.store = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.store:
            self.store[key] = []
        self.store[key].append([timestamp, value])

    def get(self, key: str, timestamp: int) -> str:
        if not key in self.store:
            return ""
        key_list = self.store[key]
        l = 0
        r = len(key_list) - 1
        while l <= r:
            mid = (l + r) // 2
            key_timestamp = key_list[mid][0]
            key_value = key_list[mid][1]
            if key_timestamp == timestamp:
                return key_value
            if key_timestamp < timestamp:
                l = mid + 1
            else:
                r = mid - 1
        if r >= 0:
            return key_list[r][1]
        return ""
