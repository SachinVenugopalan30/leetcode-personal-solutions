# intuition
# create a dequeue to and add all elements from nums
# take your first window, find the largest element in the window
# what we do next is to remove all elements from the dequeue prior to the largest
# element in the window, we do not need to check these elements ever again
# we move the window, and then check if the largest element
# in the window is greater than the element at the front of the dequeue
# if it not, add the new element to the dequeue. we only remove elements that are smaller
# than the new element, because they will never be the largest element in
# the window again

import collections


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        l = r = 0
        q = collections.deque()
        # pop smaller values from the deque
        while r < len(nums):
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)

            # remove left val from window
            if l > q[0]:
                q.popleft()

            if (r + 1) >= k:
                res.append(nums[q[0]])
                l += 1
            r += 1
        return res
