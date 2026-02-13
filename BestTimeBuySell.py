# 121. Best Time to Buy and Sell Stock
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        min_val = prices[0]

        for price in prices[1:]:
            max_profit = max(max_profit, price - min_val)
            min_val = min(min_val, price)
        return max_profit
