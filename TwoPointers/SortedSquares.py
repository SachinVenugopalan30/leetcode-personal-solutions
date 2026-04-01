class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left = 0
        right = len(nums) - 1
        result = [0] * len(nums)
        pos = len(nums) - 1  # fill result from the back
        # intuition: since the input is already sorted, largest absolute values are always at the two ends
        # place the larger one at the back of the result array, and move that pointer inward
        # we need to fill in the result array from right to left
        while left <= right:
            left_sq = nums[left] ** 2
            right_sq = nums[right] ** 2

            if left_sq > right_sq:
                result[pos] = left_sq
                left += 1
            else:
                result[pos] = right_sq
                right -= 1

            pos -= 1

        return result
