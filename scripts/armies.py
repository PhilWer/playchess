### TODO: move to config.py or create separate scripts in a meaningful way
###		  (e.g. general vs army specific)

white = {'close_squares':  	['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 
						     'a2', 'b2', 'c2', 'd2','e2', 'f2', 'g2', 'h2'
						   	],
		 'far_squares': 	['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', 
						     'a7', 'b7', 'c7', 'd7','e7', 'f7', 'g7', 'h7'
					   	  	],
		 'king_short':	{'piece': 'king_e1', 'from': 'e1', 'to': 'g1'},
		 'king_long' :	{'piece': 'king_e1', 'from': 'e1', 'to': 'c1'},
		 'rook_short':	{'piece': 'rook_h1', 'from': 'h1', 'to': 'f1'},
		 'rook_long' :	{'piece': 'rook_a1', 'from': 'a1', 'to': 'd1'}
		}

black = {'close_squares':  	['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', 
						     'a7', 'b7', 'c7', 'd7','e7', 'f7', 'g7', 'h7'
					   	  	],
		 'far_squares': 	['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 
						     'a2', 'b2', 'c2', 'd2','e2', 'f2', 'g2', 'h2'
						   	],
		 'king_short':	{'piece': 'king_e8', 'from': 'e8', 'to': 'g8'},
		 'king_long' :	{'piece': 'king_e8', 'from': 'e8', 'to': 'c8'},
		 'rook_short':	{'piece': 'rook_h8', 'from': 'h8', 'to': 'f8'},
		 'rook_long' :	{'piece': 'rook_a8', 'from': 'a8', 'to': 'd8'}
		}