class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        found = set()
        for num in nums:
            if num in found:
                return num
            found.add(num)
        return 0