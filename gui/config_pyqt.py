#!/usr/bin/env python


def chessboard_utils(army = 'white'):
    """_summary_

    Args:
        army (string): 
    
    Returns:
        dict:
    """
    if army == 'white':
        # The iterator go through (0, 1, ... 7) values
        start, stop, step = (0, 8, 1)
        ascending = True
    elif army == 'black':
        # The iterator go through (7, 6, ... 0) values
        start, stop, step = (7, -1, -1)
        ascending = False
    else:
        raise ValueError('The `army` value must be `black` or `white`. You passed ' + str(army))

    chessboard_numbers = {}                 # name of each square associated to its ordinal number
    square_names_matrix = [[None] * 8] * 8  # name of each square
    square_colors = {}                      # color of each square
    color = -1  # the first cell (either `a1` or `h8`) is black
    for i in range(start, stop, step):     # row
        for j in range(start, stop, step): # col
            # Increment row by one so that it spans [1, 8] instead of [0, 7]
            row = i + 1
            # Turn the column number into the corresponding char
            col = chr(j + ord('a'))
            # Define the cell name
            cell_name = col + str(row)
            # Add the cell name as a dict key, with the corresponding number as value
            cell_number = i * 8 + j  
            chessboard_numbers[cell_name] = cell_number if ascending else 63 - cell_number
            # Put the cell name into the names matrix
            square_names_matrix[i][j] = cell_name
            # Assign the color to the cell
            square_colors[cell_name] = 'white' if color == 1 else 'black'
            color *= (-1) # toggle color

    return chessboard_numbers, square_names_matrix, square_colors


chessboard_numbers_black, matrix, chessboard_colors_black = chessboard_utils(army = 'black')
chessboard_numbers_white, matrix, chessboard_colors_white = chessboard_utils(army = 'white')


#Pieces characteristics
king = {
'name': 'king',
'diameter': 0.03, #m
'weight': 34, #g
'height': 0.085, #m
'center_height': 0.8875, #m
'gripper_closure': 'closing_for_king'
}

queen = {
'name': 'queen',
'diameter': 0.03,
'weight': 33,
'height': 0.075,
'center_height': 0.8825,
'gripper_closure': 'closing_for_queen'
}

bishop = {
'name': 'bishop',
'diameter': 0.025,
'weight': 20,
'height': 0.064,
'center_height': 0.877,
'gripper_closure': 'closing_for_bishop'
}

knight = {
'name': 'knight',
'diameter': 0.03,
'weight': 25,
'height': 0.06,
'center_height': 0.875,
'gripper_closure': 'closing_for_knight'
}

rook = {
'name': 'rook',
'diameter': 0.025,
'weight': 18,
'height': 0.046,
'center_height': 0.868,
'gripper_closure': 'closing_for_rook'
}

pawn = {
'name': 'pawn',
'diameter': 0.025,
'weight': 15,
'height': 0.04,
'center_height': 0.865,
'gripper_closure': 'closing_for_pawn'
}

#Pieces coordinates at the start of the game
pieces_coordinates = {
'rook_h1': ['h1', rook],
'knight_g1': ['g1', knight],
'bishop_f1': ['f1', bishop],
'king_e1': ['e1', king],
'queen_d1': ['d1', queen],
'bishop_c1': ['c1', bishop],
'knight_b1': ['b1', knight],
'rook_a1': ['a1', rook],
'pawn_h2': ['h2', pawn],
'pawn_g2': ['g2', pawn],
'pawn_f2': ['f2', pawn],
'pawn_e2': ['e2', pawn],
'pawn_d2': ['d2', pawn],
'pawn_c2': ['c2', pawn],
'pawn_b2': ['b2', pawn],
'pawn_a2': ['a2', pawn],
'pawn_h7': ['h7', pawn],
'pawn_g7': ['g7', pawn],
'pawn_f7': ['f7', pawn],
'pawn_e7': ['e7', pawn],
'pawn_d7': ['d7', pawn],
'pawn_c7': ['c7', pawn],
'pawn_b7': ['b7', pawn],
'pawn_a7': ['a7', pawn],
'rook_h8': ['h8', rook],
'knight_g8': ['g8', knight],
'bishop_f8': ['f8', bishop],
'king_e8': ['e8', king],
'queen_d8': ['d8', queen],
'bishop_c8': ['c8', bishop],
'knight_b8': ['b8', knight],
'rook_a8': ['a8', rook]
}

white_pawns = ['pawn_a2', 'pawn_b2', 'pawn_c2', 'pawn_d2', 'pawn_e2', 'pawn_f2', 'pawn_g2', 'pawn_h2']

black_pawns = ['pawn_a7', 'pawn_b7', 'pawn_c7', 'pawn_d7', 'pawn_e7', 'pawn_f7', 'pawn_g7', 'pawn_h7']
