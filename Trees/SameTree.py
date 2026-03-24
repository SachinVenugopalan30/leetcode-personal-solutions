from collections import deque


class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False

        # ONE LINER RECURSIVE APPROACH - WHAT I SHOULD'VE CODED LMAO
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

        ## BFS APPROACH - WHAT I CODED
        if p and q:
            q1 = deque([p])
            q2 = deque([q])

            while q1 and q2:
                node1 = q1.popleft()
                node2 = q2.popleft()

                if not node1.left and node2.left:
                    return False
                if node1.left and not node2.left:
                    return False
                if not node1.right and node2.right:
                    return False
                if node1.right and not node2.right:
                    return False

                if node1.left and node2.left:
                    left1 = node1.left
                    left2 = node2.left
                    if left1.val != left2.val:
                        return False
                    q1.append(left1)
                    q2.append(left2)

                if node2.right and node2.right:
                    right1 = node1.right
                    right2 = node2.right
                    if right1.val != right2.val:
                        return False
                    q1.append(right1)
                    q2.append(right2)
            return True
