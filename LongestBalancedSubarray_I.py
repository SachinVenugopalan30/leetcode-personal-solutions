# 3719. Longest Balanced Subarray I
# https://leetcode.com/problems/longest-balanced-subarray-i


class Solution:
    def longestBalanced(self, nums: List[int]) -> int:
        # intuition: brute force every subarray with two pointers (l, r)
        # a subarray is "balanced" when the count of distinct even numbers equals the count of distinct odd numbers
        # for each starting index l, expand r and track distinct evens/odds using sets
        # only count a number if we haven't seen it before in the current window (duplicates don't add to the count)
        # whenever even_cnt == odd_cnt, the current subarray is balanced so update the answer
        def is_even(n):
            return n % 2 == 0

        n = len(nums)
        ans = 0
        for l in range(n):
            seen_even, seen_odd = set(), set()
            even_cnt, odd_cnt = 0, 0
            for r in range(l, n):
                # only count distinct values for each parity
                if is_even(nums[r]) and nums[r] not in seen_even:
                    seen_even.add(nums[r])
                    even_cnt += 1
                elif not is_even(nums[r]) and nums[r] not in seen_odd:
                    seen_odd.add(nums[r])
                    odd_cnt += 1
                # balanced when distinct even count matches distinct odd count
                if even_cnt == odd_cnt:
                    ans = max(ans, r - l + 1)
        return ans


## AI GENERATED RESULT USE FOR REFERENCE ONLY
class SolutionOptimized:
    def longestBalanced(self, nums: List[int]) -> int:
        # intuition: same O(n^2) enumeration but optimized with a single set and bitwise parity check
        # instead of maintaining separate sets for even/odd, use one set for all distinct values
        # and a single counter "diff" = (distinct evens - distinct odds)
        # when diff == 0, the subarray is balanced
        # use nums[r] & 1 to determine parity: 0 for even, 1 for odd
        # this avoids redundant set lookups and simplifies the bookkeeping
        n = len(nums)
        ans = 0
        for l in range(n):
            seen = set()
            diff = 0  # distinct_evens - distinct_odds
            for r in range(l, n):
                if nums[r] not in seen:
                    seen.add(nums[r])
                    # even -> +1, odd -> -1
                    diff += 1 if nums[r] % 2 == 0 else -1
                if diff == 0:
                    ans = max(ans, r - l + 1)
        return ans
