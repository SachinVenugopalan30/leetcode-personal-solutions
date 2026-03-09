# Problem 21
# https://leetcode.com/problems/merge-two-sorted-lists/


# intuition: create a dummy node to serve as our initial head of the result
# set a pointer cur to the dummy node
# loop till we reach the end of either list
# if list1's value is smaller, set cur's next to point to that, and then move up list1
# similary, do the same for list2
# at the end of each iteration, move up the cur pointer
# after the loop, we simply need to merge the remaining part of the list we didn't loop through to complete


class Solution:
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode(0)
        cur = dummy

        # Loop while both lists have nodes to compare
        while list1 and list2:
            if list1.val < list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next

        # Attach the remaining part of whichever list isn't empty
        cur.next = list1 or list2

        # Return the start of the real list
        return dummy.next
