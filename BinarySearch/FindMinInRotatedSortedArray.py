# 153. Find Minimum in Rotated Sorted Array
# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/


# intuition: we know the array was sorted, so we can immediately think of binary search
# compare the middle element with the last element and the first element
# if the middle element > last element, we know the minimum is on the right side
# set the low pointer to mid + 1 - we set it to mid+1 since we know mid cannot be the value since its already greater than the last element
# else, the element is on the left side or is the middle element itself
#
class Solution:
    def findMin(self, nums: List[int]) -> int:
        low = 0
        high = len(nums) - 1

        while low < high:
            mid = (low + high) // 2

            # Case 1: The pivot/min is in the right half
            if nums[mid] > nums[high]:
                low = mid + 1
            # Case 2: The min is mid or in the left half
            else:
                high = mid

        return nums[low]
