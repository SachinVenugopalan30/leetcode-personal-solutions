# 739. Daily Temperatures
# https://leetcode.com/problems/daily-temperatures/

# intuition: create stack to track indices
# Loop through temperatures, if current temp is greater than temp at top of stack, pop from stack
# calculate the difference in indices to get the number of days until a warmer temp.
# Push current index onto stack.
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []  # stores indices

        for i in range(n):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                j = stack.pop()
                ans[j] = i - j
            stack.append(i)

        return ans
