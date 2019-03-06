import random as rand

pieces = ['q']
pieces.extend(('r', 'n', 'b')*2)
pieces.extend(('p')*8)
pieces.extend([piece.upper() for piece in pieces])

board=[['.']*8 for space in range(8)]

while pieces:
    piece = pieces[0]
    row=rand.randrange(8)
    col=rand.randrange(8)

    if rand.randrange(3 if piece.lower() == 'p' else 5) == 0:
        pieces.remove(piece)
        continue

    board[row][col]=piece
    pieces.remove(piece)

    
row=rand.randrange(8)
col=rand.randrange(8)

row1=rand.randrange(8)
col1=rand.randrange(8)

while row==row1 and col==col1:
    row1=rand.randrange(8)
    col1=rand.randrange(8)


board[row][col] = 'k'
board[row1][col1] = 'K'

board = [''.join(row) for row in board]
    
print(board)

with open("board.txt","w") as f:
    for row in board:        
        f.write(row + '\n')
