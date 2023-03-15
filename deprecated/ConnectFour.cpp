#include <iostream>

class Board {

    private:
        const int ROWS = 6;
        const int COLS = 7;

    public:
        int board[this->ROWS][this->COLS] = {};
        
        Board() {

        };

        /*
          0 1 2 3 4 5 6
        5 - - - - - - -
        4 - - - - - - -
        3 - - - - - - -
        2 - - - - - - -
        1 - - - - - - -
        0 - - - - - - -
        */

        bool checkWinner(int row, int col) {
            int ID = board[row][col];

            // check winner
            return countDirection(row, col + 1, 0, 1, ID) + countDirection(row, col - 1, 0, -1, ID) >= 4
                || countDirection(row + 1, col, 1, 0, ID) + countDirection(row - 1, col, -1, 0, ID) >= 4
                || countDirection(row + 1, col + 1, 1, 1, ID) + countDirection(row - 1, col - 1, -1, -1, ID) >= 4
                || countDirection(row + 1, col - 1, 1, -1, ID) + countDirection(row - 1, col + 1, -1, 1, ID) >= 4;



        };

        // recursive function to count the number of pieces in a row in a direction
        int countDirection(int row, int col, int rowDelta, int colDelta, int ID) {
            if (row < 0 || row >= ROWS || col < 0 || col >= COLS) {
                return 0;
            }
            
            if (board[row][col] == ID) {
                return 1 + countDirection(row + rowDelta, col + colDelta, rowDelta, colDelta, ID);
            }

            return 0;
        }

        bool placePiece(int c, int ID) {
            // check if the column is full already
            if (board[0][c] != 0) {
                return false;
            }

            for (int r = ROWS - 1; r >= 0; r--) {
                if (board[r][c] == 0) {
                    board[r][c] = ID;
                    if (checkWinner(r, c)) {
                        return true;
                    }

                    return false;
                }
            }
            return false;
        };

        void outputToNetwork() {
            for (int r = 0; r < ROWS; r++) {
                for (int c = 0; c < COLS; c++) {
                    
                }
            }
        }
        
};

int main(char argv, char[] argv) {

}