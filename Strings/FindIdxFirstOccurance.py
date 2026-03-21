# Leetcode Prob. 28
# https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if haystack == needle:
            return 0
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i] != needle[0]:
                continue
            start = i
            end = i + len(needle)
            if haystack[start:end] == needle:
                return i
        return -1
