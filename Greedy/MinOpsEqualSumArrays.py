# 1775. Equal Sum Arrays with Minimum Number of Operations
# https://leetcode.com/problems/equal-sum-arrays-with-minimum-number-of-operations/description/


class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        sum1, sum2 = sum(nums1), sum(nums2)

        # ensure sum1 <= sum2 (so we always increase sum1 or decrease sum2)
        if sum1 > sum2:
            return self.minOperations(nums2, nums1)

        # impossible check
        # maximum possible sum of the smaller array is still less than the minimum possible sum of the larger array
        if len(nums1) * 6 < len(nums2) * 1:
            return -1

        # sort nums1 ascending (best gains at front)
        # sort nums2 descending (best gains at front)
        nums1.sort()
        nums2.sort(reverse=True)

        gap = sum2 - sum1
        ops = 0
        p1, p2 = 0, 0

        while gap > 0:
            # gain from increasing nums1[p1] to 6
            # gain from decreasing nums2[p2] to 1
            gain1 = 6 - nums1[p1] if p1 < len(nums1) else 0
            gain2 = nums2[p2] - 1 if p2 < len(nums2) else 0

            if gain1 >= gain2:
                gap -= gain1
                p1 += 1
            else:
                gap -= gain2
                p2 += 1

            ops += 1

        return ops
