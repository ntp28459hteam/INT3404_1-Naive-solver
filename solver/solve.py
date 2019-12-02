import sys
import numpy as np

CELL_WIDTH = 3
CELL_HEIGHT = 3
WIDTH = 3
HEIGHT = 3
BOARD_WIDTH = CELL_WIDTH * WIDTH
BOARD_HEIGHT = CELL_HEIGHT * HEIGHT
MAX = CELL_WIDTH * CELL_HEIGHT

# board[BOARD_HEIGHT][BOARD_WIDTH]
# given_val[BOARD_HEIGHT][BOARD_WIDTH]

# col[BOARD_WIDTH][MAX]
# row[BOARD_HEIGHT][MAX]
# cell[HEIGHT][WIDTH][MAX]

board = []
# board.append([])
given_val = []
col = []
row = []
cell = []

def mock_test():
    solution_file_path = './sample_puzzle.sol'
    f = open(solution_file_path, "r")
    lines = []
    line = f.readline()
    while line:
        lines.append(line)
        line = f.readline()
    f.close()

    puzzle_file_path = './sample_puzzle.dat'
    f2 = open(puzzle_file_path, "r")
    lines2 = []
    line2 = f2.readline()
    while line2:
        lines2.append(line2)
        line2 = f2.readline()
    f2.close()

    puzzle_state = []

    for line, line2 in zip(lines, lines2):
        line = line.strip()
        line2 = line2.strip()
        # for filled, digit_str in zip(line2.split(' '), line.split(' ')):
        # 	print (filled, digit_str)
        rows = [(int(filled), int(digit_str)) for filled, digit_str in zip(line2.split(' '), line.split(' '))]
        # puzzle_state.extend(rows)
        puzzle_state.append(rows)
    return puzzle_state

def solve_puzzle(board_):
    # print (np.shape(board))
    board = board_.copy()
    # print (np.shape(solve.board))
    # f = open("../sample_puzzle.dat", "r")
    
    # # fout = open("../sample_puzzle.sol", "w")
    sys.stdout = open('../sample_puzzle_debug.sol', 'w')

    # line = f.readline()

    for i in range(BOARD_HEIGHT):
        given_val_ = []
        col_ = []
        row_ = []
        cell_ = []
        for j in range(BOARD_WIDTH):
            given_val_.append(False)
            col_.append(False)
            row_.append(False)

            cell__ = []
            for k in range(MAX):
                cell__.append(False)
            cell_.append(cell__)
        given_val.append(given_val_)
        col.append(col_)
        row.append(row_)
        cell.append(cell_)

    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            if(board[i][j] != 0):
                given_val[i][j] = True
                col[j][board[i][j] - 1] = False
                row[i][board[i][j] - 1] = False
                cell[int(i / CELL_HEIGHT)][int(j / CELL_WIDTH)][board[i][j] - 1] = True
            
    # f.close()
    solve(0,0)

def solve(y, x):
    print (np.shape(board))
    if(given_val[y][x] == True):
        if(x == BOARD_WIDTH - 1):
            solve(y + 1, 0)
        else:
            solve(y, x + 1)

    if(y == BOARD_HEIGHT - 1 and x == BOARD_WIDTH - 1):
        print_sol()
        return

    # int i; // what will happen if we use global variable instead!?
    for i in range(1, MAX + 1):
        if(True
           and not given_val[y][x] # not filled
           and col[x][i - 1] == 0
           and row[y][i - 1] == 0
           and cell[int(y / CELL_HEIGHT)][int(x / CELL_WIDTH)][i - 1] == 0
           and True
           ):
            col[x][i - 1] = 1
            row[y][i - 1] = 1
            cell[int(y / CELL_HEIGHT)][int(x / CELL_WIDTH)][i - 1] = 1

            board[y][x] = i
            if(y == BOARD_HEIGHT - 1 and x == BOARD_WIDTH - 1):
                print_sol()
                return
            if(x == BOARD_WIDTH - 1):
                solve(y + 1, 0)
            else:
                solve(y, x + 1)

            board[y][x] = 0
            col[x][i - 1] = 0
            row[y][i - 1] = 0
            cell[int(y / CELL_HEIGHT)][int(x / CELL_WIDTH)][i - 1] = False

def print_sol():
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            print(board[i][j])
        print('', end='\n')

