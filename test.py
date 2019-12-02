import cv2
import numpy as np
import preprocess
import solver

from nn import NeuralNetwork

original = preprocess.original_image
cv2.imshow('original', original)
processed = preprocess.preprocess_image(original)
cv2.imshow('processed', processed)
corners = preprocess.find_corners(processed)
cropped, projection_mat = preprocess.crop_and_warp(original, corners)

squares = preprocess.infer_grid(cropped)
digits = preprocess.get_digits(cropped, squares, 28)

digits_ = []

for i in range(9):
    for j in range(9):
        # row_ = []
        digit = digits[9 * j + i].copy()
        digits_.append(digit)
        
out, tiles = preprocess.show_digits(digits_, show=True, save=True)  # cv2.imshow('digit', digits[1])

'''
Solver
Method being under development with unhandled recursion, please do not run
'''
# solver.solver(solver.solve.mock_test())

puzzle_state = solver.solver(solver.solve.mock_test(), solve_bool=False)
# puzzle_state = solver.solve.mock_test()
print (np.shape(puzzle_state))

'''
Digit recognition
'''

# being development 
# for i in range(81):
#     tile = digits[i] # tile = tiles[i]
#     preprocess.guess_number(tile, i)
# guy = NeuralNetwork.instance()
# prediction = guy.guess(digits[12])
# number = np.argmax(prediction, axis=0)
# print (number)

'''
Digit interpretation
'''

# for i, square in enumerate(squares):
#     scale = int((squares[0][1][0] - squares[0][0][0]) * 0.073) # scale = int ((cropped / 9 - 15 * 2) * 0.075)
#     fh, fw = cv2.getTextSize(str(puzzle_state[i][1]), cv2.FONT_HERSHEY_PLAIN, scale, thickness =3)[0]  # Get font height and width
#     h_pad = int((square[1][0] - square[0][0] - fw) / 2)  # Padding to centre the number
#     v_pad = int((square[1][1] - square[0][1] - fw) / 2)
#     h_pad -= 15  # No border on the original, so this compensates
#     v_pad += 15
#     if(puzzle_state[i][0] == 0): # not filled
#         cropped = cv2.putText(cropped, str(puzzle_state[i][1]), (int(square[0][0]) + h_pad, int(square[1][1]) - v_pad),
#             cv2.FONT_HERSHEY_PLAIN, fontScale=scale, color=(50, 50, 255), thickness= 3)

#     height, width = original.shape[:2]
#     img = cv2.warpPerspective(cropped, projection_mat, (width, height), flags=cv2.WARP_INVERSE_MAP, dst=original,
#                                 borderMode=cv2.BORDER_TRANSPARENT)
cv2.imwrite('./debug/cropped.jpg', cropped)
cv2.imwrite('./debug/out.jpg', out)
cv2.waitKey(0)