class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # clean solution
        average = sum(nums[:k]) / k
        current_sum = sum(nums[:k])

        for i in range(1, len(nums) - k + 1):
            current_sum += nums[i + k - 1] - nums[i - 1]  # slide the window
            average = max(average, current_sum / k)

        return average

        # my solution - sum is recomputed at O(k) each iteration causing TLE
        # clean solution above computes sum at O(1) by just removing the previous element, adding the next element (moving the window)
        if len(nums) == 1:
            return nums[0]
        if not len(nums):
            return 0
        average = sum(nums[:k]) / k
        for i in range(len(nums) - k + 1):
            sub_arr = nums[i : i + k]
            arr_avg = sum(sub_arr) / k
            average = max(arr_avg, average)
        return average
