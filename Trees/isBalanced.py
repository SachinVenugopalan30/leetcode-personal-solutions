class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(node):
            if not node:
                return 0

            left = dfs(node.left)
            right = dfs(node.right)

            # Propagate imbalance up the tree
            if left == -1 or right == -1:
                return -1

            # If difference > 1, signal imbalance
            if abs(left - right) > 1:
                return -1

            return 1 + max(left, right)

        return dfs(root) != -1
