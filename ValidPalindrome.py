# Problem 125 - Valid Palindrome
#  https://leetcode.com/problems/valid-palindrome/description/


class Solution:
    def isPalindrome(self, s: str) -> bool:
        end = len(s) - 1
        start = 0
        while start < end:
            # ignore special characters and only move that pointer
            if not s[start].isalnum():
                start += 1
                continue
            elif not s[end].isalnum():
                end -= 1
                continue
            if s[start].lower() != s[end].lower():
                return False
            start += 1
            end -= 1
        return True
