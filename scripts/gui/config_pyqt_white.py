#!/usr/bin/env python

#Numbers associated with each square

chessboard_numbers = {
'a1': 0,
'b1': 1,
'c1': 2,
'd1': 3,
'e1': 4,
'f1': 5,
'g1': 6,
'h1': 7,
'a2': 8,
'b2': 9,
'c2': 10,
'd2': 11,
'e2': 12,
'f2': 13,
'g2': 14,
'h2': 15,
'a3': 16,
'b3': 17,
'c3': 18,
'd3': 19,
'e3': 20,
'f3': 21,
'g3': 22,
'h3': 23,
'a4': 24,
'b4': 25,
'c4': 26,
'd4': 27,
'e4': 28,
'f4': 29,
'g4': 30,
'h4': 31,
'a5': 32,
'b5': 33,
'c5': 34,
'd5': 35,
'e5': 36,
'f5': 37,
'g5': 38,
'h5': 39,
'a6': 40,
'b6': 41,
'c6': 42,
'd6': 43,
'e6': 44,
'f6': 45,
'g6': 46,
'h6': 47,
'a7': 48,
'b7': 49,
'c7': 50,
'd7': 51,
'e7': 52,
'f7': 53,
'g7': 54,
'h7': 55,
'a8': 56,
'b8': 57,
'c8': 58,
'd8': 59,
'e8': 60,
'f8': 61,
'g8': 62,
'h8': 63
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
'''
matrix = [
['self.a8', 'self.b8', 'self.c8', 'self.d8', 'self.e8', 'self.f8', 'self.g8', 'self.h8'],
['self.a7', 'self.b7', 'self.c7', 'self.d7', 'self.e7', 'self.f7', 'self.g7', 'self.h7'],
['self.a6', 'self.b6', 'self.c6', 'self.d6', 'self.e6', 'self.f6', 'self.g6', 'self.h6'],
['self.a5', 'self.b5', 'self.c5', 'self.d5', 'self.e5', 'self.f5', 'self.g5', 'self.h5'],
['self.a4', 'self.b4', 'self.c4', 'self.d4', 'self.e4', 'self.f4', 'self.g4', 'self.h4'],
['self.a3', 'self.b3', 'self.c3', 'self.d3', 'self.e3', 'self.f3', 'self.g3', 'self.h3'],
['self.a2', 'self.b2', 'self.c2', 'self.d2', 'self.e2', 'self.f2', 'self.g2', 'self.h2'],
['self.a1', 'self.b1', 'self.c1', 'self.d1', 'self.e1', 'self.f1', 'self.g1', 'self.h1']
]
'''
'''
matrix = [
['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'],
['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'],
['a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6'],
['a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5'],
['a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4'],
['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3'],
['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'],
['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']
]
'''
matrix = [
['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1'],
['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'],
['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3'],
['a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4'],
['a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5'],
['a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6'],
['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'],
['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8']
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

