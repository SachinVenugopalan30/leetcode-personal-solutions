# 3. Longest Substring Without Repeating Characters
# https://leetcode.com/problems/longest-substring-without-repeating-characters/


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # intuition: we iterate through the string and keep track of the characters
        # we have seen in a set. If we see a character that is already in the set,
        # we remove characters from the left until we remove the duplicate character.
        # We also keep track of the maximum length of the substring
        # we have seen so far.
        charSet = set()
        res = 0
        l = 0
        for r in range(len(s)):
            # if right character is seen in the set
            # remove left characters until the right character is removed
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1
            # add the newest right character back to the set
            charSet.add(s[r])
            # update the result with the maximum length of the substring seen so far
            res = max(res, r - l + 1)
        return res
