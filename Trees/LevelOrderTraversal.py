from collections import deque


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            tmp = []
            # we iterate through the queue, which currently contains all the elements in that level to append to the result.
            for _ in range(len(q)):
                node = q.popleft()
                if node:
                    tmp.append(node.val)
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
            res.append(tmp)
        return res
