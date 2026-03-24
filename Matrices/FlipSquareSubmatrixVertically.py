class Solution:
    def reverseSubmatrix(
        self, grid: List[List[int]], x: int, y: int, k: int
    ) -> List[List[int]]:
        # two pointer approach, top pointer starts from the top of the given square x,
        # bottom pointer starts from the bottom of the given square x + k - 1
        i = x
        j = x + k - 1
        while i < j:
            for l in range(y, y + k):
                grid[i][l], grid[j][l] = grid[j][l], grid[i][l]
            i += 1
            j -= 1
        return grid
