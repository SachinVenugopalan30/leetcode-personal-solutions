class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        seen = set()

        while head:
            if head not in seen:
                seen.add(head)
                if head.next in seen:
                    return True
                head = head.next
            else:
                return True
        return False
