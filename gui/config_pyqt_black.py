#!/usr/bin/env python

#Numbers associated with each square

chessboard_numbers = {
'h8': 0,
'g8': 1,
'f8': 2,
'e8': 3,
'd8': 4,
'c8': 5,
'b8': 6,
'a8': 7,
'h7': 8,
'g7': 9,
'f7': 10,
'e7': 11,
'd7': 12,
'c7': 13,
'b7': 14,
'a7': 15,
'h6': 16,
'g6': 17,
'f6': 18,
'e6': 19,
'd6': 20,
'c6': 21,
'b6': 22,
'a6': 23,
'h5': 24,
'g5': 25,
'f5': 26,
'e5': 27,
'd5': 28,
'c5': 29,
'b5': 30,
'a5': 31,
'h4': 32,
'g4': 33,
'f4': 34,
'e4': 35,
'd4': 36,
'c4': 37,
'b4': 38,
'a4': 39,
'h3': 40,
'g3': 41,
'f3': 42,
'e3': 43,
'd3': 44,
'c3': 45,
'b3': 46,
'a3': 47,
'h2': 48,
'g2': 49,
'f2': 50,
'e2': 51,
'd2': 52,
'c2': 53,
'b2': 54,
'a2': 55,
'h1': 56,
'g1': 57,
'f1': 58,
'e1': 59,
'd1': 60,
'c1': 61,
'b1': 62,
'a1': 63
}

chessboard_colors = {
'a1': 'black',
'b1': 'white',
'c1': 'black',
'd1': 'white',
'e1': 'black',
'f1': 'white',
'g1': 'black',
'h1': 'white',
'a2': 'white',
'b2': 'black',
'c2': 'white',
'd2': 'black',
'e2': 'white',
'f2': 'black',
'g2': 'white',
'h2': 'black',
'a3': 'black',
'b3': 'white',
'c3': 'black',
'd3': 'white',
'e3': 'black',
'f3': 'white',
'g3': 'black',
'h3': 'white',
'a4': 'white',
'b4': 'black',
'c4': 'white',
'd4': 'black',
'e4': 'white',
'f4': 'black',
'g4': 'white',
'h4': 'black',
'a5': 'black',
'b5': 'white',
'c5': 'black',
'd5': 'white',
'e5': 'black',
'f5': 'white',
'g5': 'black',
'h5': 'white',
'a6': 'white',
'b6': 'black',
'c6': 'white',
'd6': 'black',
'e6': 'white',
'f6': 'black',
'g6': 'white',
'h6': 'black',
'a7': 'black',
'b7': 'white',
'c7': 'black',
'd7': 'white',
'e7': 'black',
'f7': 'white',
'g7': 'black',
'h7': 'white',
'a8': 'white',
'b8': 'black',
'c8': 'white',
'd8': 'black',
'e8': 'white',
'f8': 'black',
'g8': 'white',
'h8': 'black'
}



#Matrix definition of the squares
matrix = [
['h8', 'g8', 'f8', 'e8', 'd8', 'c8', 'b8', 'a8'],
['h7', 'g7', 'f7', 'e7', 'd7', 'c7', 'b7', 'a7'],
['h6', 'g6', 'f6', 'e6', 'd6', 'c6', 'b6', 'a6'],
['h5', 'g5', 'f5', 'e5', 'd5', 'c5', 'b5', 'a5'],
['h4', 'g4', 'f4', 'e4', 'd4', 'c4', 'b4', 'a4'],
['h3', 'g3', 'f3', 'e3', 'd3', 'c3', 'b3', 'a3'],
['h2', 'g2', 'f2', 'e2', 'd2', 'c2', 'b2', 'a2'],
['h1', 'g1', 'f1', 'e1', 'd1', 'c1', 'b1', 'a1']
]

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

