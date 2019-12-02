import numpy as np
from nn import NeuralNetwork
from numpy_ringbuffer import RingBuffer

import solver.solve

def guess_number(tile, i, puzzle_state, kind = 0):
    if kind == 0:
        if tile is None:
            number = 0
        else:
            guy = NeuralNetwork.instance()
            prediction = guy.guess(tile)
            number = np.argmax(prediction, axis=0)

        puzzle_state[i] = number

    else:
        # Being development
        None
        # prev_guesses = RingBuffer(capacity=5, dtype=(float, (10)))
        # timer = 9
        # maxtimer = 10
        # confidence_threshold = 0
        # if timer >= maxtimer:
        #     timer = 0

        #     if tile is None:
        #         prev_guesses.appendleft(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        #     else:
        #         guy = NeuralNetwork.instance()
        #         prediction = guy.guess(tile)
        #         prev_guesses.appendleft(np.array(prediction))

        # m = np.mean(prev_guesses, axis=0)
        # number = np.argmax(m, axis=0)
        # if m[number] > confidence_threshold:
        #     puzzle_state[i] = number

def solver(puzzle_state, solve_bool=False):
    board = []
    for row in puzzle_state:
        # print (row)
        new_row = []
        for digit_tuple in row:
            if digit_tuple[0] == 0:
                new_row.append(0)
            else:
                new_row.append(digit_tuple[1])
        board.append(new_row)
    # print (board)
    solve.board = board
    # if solve:
        # solve.solve_puzzle(board)
    return board

def print_puzzle_state(puzzle_state):
    for i in range (9):
        for j in range(9):
            print (puzzle_state[i * 9 + j], end = ' ') # alternative : print ("%d") ...
        print (end = '\n') # print ("\n") makes two newline character