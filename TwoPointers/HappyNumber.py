class Solution:
    def isHappy(self, n: int) -> bool:
        def getNext(n):
            return sum(int(d) ** 2 for d in str(n))

        slow = getNext(n)
        fast = getNext(getNext(n))
        while slow != fast:
            slow = getNext(slow)
            fast = getNext(getNext(fast))
        return slow == 1
