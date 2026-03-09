# 11. Container With Most Water
# https://leetcode.com/problems/container-with-most-water/


# intuition: 2 pointer approach, we start with the left and right pointer at the ends of the array
# we calculate the area between the two pointers, and then move the pointer that has the smaller height inward,
# since the area is limited by the shorter line
# we continue this process until the two pointers meet, keeping track of the maximum area found.
class Solution:
    def maxArea(self, heights: List[int]) -> int:
        left = 0
        right = len(heights) - 1
        max_area = min(heights[left], heights[right]) * (right - left)
        while left < right:
            new_area = min(heights[left], heights[right]) * (right - left)
            max_area = max(max_area, new_area)
            if heights[left] < heights[right]:
                left += 1
            else:
                right -= 1
        return max_area
