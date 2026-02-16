# 347. Top K Frequent Elements
# https://leetcode.com/problems/top-k-frequent-elements/


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # get and store frequency of all elements in nums
        freq = {}
        for x in nums:
            freq[x] = freq.get(x, 0) + 1

        # create buckets for each frequency
        # intuition: the index is technically the number of times the element occurs in nums
        # since each element in buckets starts as an empty list, just append the element from nums that occurs *index* number of times
        # for example, if 1 occurs 3 times in nums, then we append 1 to buckets[3]
        buckets = [[] for _ in range(len(nums) + 1)]

        for num, count in freq.items():
            buckets[count].append(num)

        # iterate backwards k times to get the k most frequent elements
        # since buckets is indexed by frequency, it is already sorted in ascending order from 0 length of nums since there
        # can be at most len(nums) frequency of an element in nums
        res = []
        for count in range(len(buckets) - 1, 0, -1):
            for num in buckets[count]:
                res.append(num)
                if len(res) == k:
                    return res
