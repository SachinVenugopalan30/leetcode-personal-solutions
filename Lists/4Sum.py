# 18. 4Sum


# intuition: exact same as 3sum and 2sum on a sorted array, fix i and j and iterate
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums) - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums) - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                start = j + 1
                end = len(nums) - 1
                while start < end:
                    total = nums[i] + nums[j] + nums[start] + nums[end]
                    if total < target:
                        start += 1
                    elif total > target:
                        end -= 1
                    else:
                        result.append([nums[i], nums[j], nums[start], nums[end]])
                        start += 1
                        end -= 1
                        while start < end and nums[start] == nums[start - 1]:
                            start += 1
                        while end > start and nums[end] == nums[end + 1]:
                            end -= 1
        return result
