import numpy as np


class Board:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.board = [[0 for i in range(cols)] 
                    for i in range(rows)]



    def checkWinner(self, r, c):
        ID = self.board[r][c]

        return self.countDirection(r, c + 1, 0, 1, ID) + self.countDirection(r, c - 1, 0, -1, ID) >= 4 \
            or self.countDirection(r + 1, c, 1, 0, ID) + self.countDirection(r - 1, c, -1, 0, ID) >= 4 \
            or self.countDirection(r + 1, c + 1, 1, 1, ID) + self.countDirection(r - 1, c - 1, -1, -1, ID) >= 4 \
            or self.countDirection(r + 1, c - 1, 1, -1, ID) + self.countDirection(r - 1, c + 1, -1, 1, ID) >= 4


    def countDirection(self, r, c, rd, cd, ID):
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return 0
        
        if (self.board[r][c] == ID):
            return 1 + self.countDirection(r + rd, c + cd, ID)
        
        return 0


    def placePiece(self, c, ID):
        if self.board[0][c] != 0:
            return

        for r in range(self.rows - 1, 0, -1):
            if self.board[r][c] == 0:
                self.board[r][c] = ID

                if self.checkWinner(r, c):
                    return True
                
                return False
        return False

    def getState(self):
        return np.matrix([(((
                1 if pos == n else 0)
                for n in range(3))
                for pos in row)
                for row in self.board])


    def getReward(self, ID):
        pass

