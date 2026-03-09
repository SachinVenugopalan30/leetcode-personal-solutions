# 36. Valid Sudoku
# https://leetcode.com/problems/valid-sudoku/


# its given we can skip those values which are just .
# intuition: finding out what is in each row and column is pretty easy, the trickier bit is finding out whats in each box
# to find out what box each element belongs to, we use the formula (r//3) * 3 + (c//3) to get the box index
# using zero indexed values, for ex. row = 4, col = 3 => box index = (4//3) * 3 + (3//3) = 1*3 + 1 = 4, which is the middle box (5th box)
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue

                if board[r][c] not in rows[r]:
                    rows[r].add(board[r][c])
                else:
                    return False

                if board[r][c] not in cols[c]:
                    cols[c].add(board[r][c])
                else:
                    return False

                box_idx = int((r // 3) * 3 + (c // 3))
                if board[r][c] not in boxes[box_idx]:
                    boxes[box_idx].add(board[r][c])
                else:
                    return False
        return True
