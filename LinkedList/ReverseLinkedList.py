# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# 206. Reverse Linked List
# https://leetcode.com/problems/reverse-linked-list/


# Logic: Save the next node, flip the current link backward,
# then slide 'prev' and 'curr' one step forward.
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head

        while curr:
            # 1. Save the next node so we don't lose it
            nxt = curr.next
            # 2. Reverse the individual link
            curr.next = prev
            # 3. Move prev forward to the current node
            prev = curr
            # 4. Move curr forward to the saved next node
            curr = nxt

        # prev is now the new head of the reversed list
        return prev
