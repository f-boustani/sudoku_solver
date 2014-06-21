import time

def cross(A,B):
    "cross product of elements in A and B"
    return [a+b for a in A for b in B]

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows,cols)
unit_list = ([cross(rows,c) for c in cols]+
             [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])

units = dict((s, [u for u in unit_list if s in u]) 
             for s in squares)
peers = dict((s, set(sum(units[s],[]))-set([s]))
             for s in squares)


#################################################################

def solve(grid):

    return search(parse_grid(grid))

#################################################################

def get_board(board):

    "Convert board into a dict of {square: digit}"
    
    return dict(zip(squares,board))


#################################################################

def parse_grid(board):
    
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    
    values = dict((s, digits) for s in squares)
    for s,d in get_board(board).items():
        if d in digits and not assign(values, s, d):
            return False 
    return values

###########################################################################

def assign(values, s, d):
    
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False

############################################################################

def eliminate(values, s, d):
    
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    
    if d not in values[s]:
        return values 
    values[s] = values[s].replace(d,'')
     
    if len(values[s]) == 0:
        return False
    
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False

    for u in units[s]:
          
        dplaces = [s for s in u if d in values[s]]
        
        if len(dplaces) == 0:
            return False 
        elif len(dplaces) == 1:
            
            if not assign(values, dplaces[0], d):
                return False
    return values


############################################################################

def search(values):
    "Using depth-first search and propagation, try all possible values."
    
    if values is False:
        return False 
    if all(len(values[s]) == 1 for s in squares):
        return values ## Solved!
    
    n,s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d))
                for d in values[s])

########################################################################
def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: return e
    return False

