# 283. Move Zeroes
# https://leetcode.com/problems/move-zeroes/


# Intuition: 2 pointer approach, we use a fast and slow pointer
# We loop with the fast pointer, if we encounter a non zero,
# Swap with the slow pointer index, move slow pointer up by 1.
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
