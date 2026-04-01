class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        if head.next is None:
            return None

        slow = head
        fast = head
        # Phase 1: detect if a cycle exists
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if (
                slow == fast
            ):  # compare references, DO NOT COMPARE not values, chances are there are multiple nodes with the same value
                break
        else:
            return None  # no cycle found

        # Phase 2: find the cycle entry point
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next

        return slow
