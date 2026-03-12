# 143. Reorder List
# https://leetcode.com/problems/reorder-list/


# intuition: break the problem into 3 sub problems
# 1. find the middle of the list using the fast-slow pointer technique
# 2. reverse the second half of the list
# 3. weave the two halves together
#   3.1. You have two lists moving "toward each other"
#   3.2. At each step, splice second's head between first and first.next
#   3.3. Stop when second has no next — the tail is already in place
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # Step 1: Find middle
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # Step 2: Reverse second half
        prev, curr = None, slow
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node

        # Step 3: Weave
        first, second = head, prev
        while second.next:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2
