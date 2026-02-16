# 15. 3Sum
# https://leetcode.com/problems/3sum/description/


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        n = len(nums)

        for i in range(n):
            # 1) stop early (optional but nice)
            if nums[i] > 0:
                break

            # 2) skip duplicate anchors
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            l, r = i + 1, n - 1

            while l < r:
                s = nums[i] + nums[l] + nums[r]

                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])

                    # move both pointers once
                    l += 1
                    r -= 1

                    # 3) skip duplicate l values
                    # Keep moving l while it’s still pointing to the same value we already used.
                    # Intuition - we already know this will add to the same sum, so we can skip it.
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1

                    # 4) skip duplicate r values
                    # Same thing as above
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1

        return res
