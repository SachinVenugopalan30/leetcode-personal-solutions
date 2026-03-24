class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


# im gonna be honest, i struggled to do this problem even though its so simple.
# just goes to show i have a long road ahead of me before i can even consider myself competent.
