class Solution:
    # despite how it looks time complexity is O(n), increment right once, subtract left once each iteration, so total is O(2n) or O(n)
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left, right = 0, 0
        current_sum = 0
        min_len = float("inf")
        for right in range(len(nums)):
            current_sum += nums[right]
            while current_sum >= target:
                min_len = min(min_len, right - left + 1)
                current_sum -= nums[left]
                left += 1
        return min_len if min_len != float("inf") else 0
