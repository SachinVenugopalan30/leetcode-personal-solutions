# 2. Add Two Numbers
# https://leetcode.com/problems/add-two-numbers/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode()
        head = dummy
        carry = 0

        while l1 or l2:
            # get values, default to 0 if list is exhausted
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0

            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10

            head.next = ListNode(digit)
            head = head.next

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        if carry:
            head.next = ListNode(carry)
        return dummy.next
