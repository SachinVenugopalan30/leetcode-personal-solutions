# 15. 3Sum
# https://leetcode.com/problems/3sum/
#
#


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        # intuition: basically solving 2sum in sorted array with the 2ptr approach by keeping the 3rd number constant
        for i in range(len(nums) - 2):
            # we do this to skip duplicates
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            start = i + 1
            end = len(nums) - 1
            while start < end:
                total = nums[i] + nums[start] + nums[end]
                if total < 0:
                    start += 1
                elif total > 0:
                    end -= 1
                else:
                    res.append([nums[i], nums[start], nums[end]])
                    start += 1
                    end -= 1
                    # keep incrementing start to skip duplicates to avoid duplicates in the final array
                    while start < end and nums[start] == nums[start - 1]:
                        start += 1
                    # keep decrementing end to skip duplicates to avoid duplicates in the final array
                    while start < end and nums[end] == nums[end + 1]:
                        end -= 1
        return res
