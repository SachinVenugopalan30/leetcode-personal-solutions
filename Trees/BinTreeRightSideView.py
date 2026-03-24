from collections import deque

# The key insight is that by snapshotting level_size like its done in level order traversal
# you know exactly when you're at the last node of a level, when i == level_size - 1.
# That last node is what's visible from the right side.


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []

        queue = deque([root])

        while queue:
            level_size = len(queue)

            for i in range(level_size):
                node = queue.popleft()

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

                # Only append the last node of each level
                if i == level_size - 1:
                    result.append(node.val)

        return result
