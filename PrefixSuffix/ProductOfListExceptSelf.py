# 238. Product of Array Except Self
# https://leetcode.com/problems/product-of-array-except-self/


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # intuition: we need to find the product of the elements to the left of the current element
        # times the product of the elements to the right of the current element
        # 2 passes of len(nums) to get the left product first, then the right product
        # keep in mind the prefix and postfix products
        res = [1] * len(nums)

        # left products
        left = 1
        for i in range(len(nums)):
            res[i] = left
            left *= nums[i]

        # right products
        right = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= right
            right *= nums[i]

        return res
