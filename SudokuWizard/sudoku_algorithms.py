import random
from random import sample
import pandas as pd
import numpy as np
import math

"""# EXPAND

GENERIC EXPANSION FUNCTION
"""

import numpy as np
def expand (node, problem):
    new_nodes = []
    possible_actions = problem.actions
    for action in possible_actions:
        if problem.is_applicable(node["state"], action):
            new_state= problem.effect (node["state"], action)
            new_node = {}
            new_node["state"]=new_state
            new_node["parent_node"]=node
            new_node["actions"]=node["actions"] + [action]
            new_node["cost"]=problem.get_cost(action, new_state)
            new_node["depth"]=node["depth"]+1
            new_node["evaluation"]=problem.get_evaluation(new_state)
            new_nodes.append (new_node)
    return new_nodes

"""# METHODS TO SOLVE

## BFS

"""

def BFS(problem):
    # result dictionary
    result = {"method":"BFS", "final_state":[], "status":"No nodes in the frontier. No solution possible.",
             "max_frontier":0, "max_depth":0, "iterations":0}

    # 1. problem definition
    # problem = Problem()
    initial_node={"state":problem.initial_state, "parent_node":{}, "actions":[], "cost":0, "depth":0, "evaluation":1}
    frontier = []

     # 2. add node to frontier
    frontier.append(initial_node)

    # 3. start exploring and expanding the frontier
    iterations=1
    while len (frontier)>0: #if we have elements in the frontier...
         # 3.1. get first element of frontier and delete it
        node = frontier[0]


        frontier = frontier[1:]
        # 3.2 check if it is final state:
        if problem.is_final_state (node["state"]):
            result["status"]="Solution Found."
            break #we end while. state will remain this last state computed, and sequence of actions will have all states.

        # 3.3 if it is not final, expand and add to the frontier
        new_nodes = expand(node, problem)
        for n in new_nodes:
            frontier.append(n)

        # we compute the maximum size of frontier: the previous one or the current if it is bigger
        result["max_frontier"]=max(result["max_frontier"],len (frontier))
         # we compute the maximum depth: the previous one or the current if it is bigger
        result["max_depth"]=max(result["max_depth"], node["depth"])
         # we update the iterations count
        result["iterations"]= iterations

        iterations+=1
      #loop keeps running until no more nodes available or final state obtained

    result["final_state"] = node
    return(result)

def BFS_g(problem):
    # result dictionary
    result = {"method":"BFS_g", "final_state":[], "status":"No nodes in the frontier. No solution possible.",
             "max_frontier":0, "max_depth":0, "iterations":0}

    # 1. problem definition
    initial_node={"state":problem.initial_state, "parent_node":{}, "actions":[], "cost":0, "depth":0, "evaluation":1}
    frontier = []

    ####
    expanded = []

     # 2. add node to frontier
    frontier.append(initial_node)

    # 3. start exploring and expanding the frontier
    iterations=1
    while len (frontier)>0: #if we have elements in the frontier...
         # 3.1. get first element of frontier and delete it
        node = frontier[0]
        frontier = frontier[1:]

        print ("---------------------------- iteration "+str(iterations)+" frontier size: "+str(len(frontier))+"\n PRocessing: ")
        print (node["state"])


        #add to expanded--> we add the state, as the rest of fields will be different
        expanded.append(node["state"])

        # 3.2 check if it is final state:
        if problem.is_final_state (node["state"]):
            result["status"]="Solution Found."
            break #we end while. state will remain this last state computed, and sequence of actions will have all states.

        # 3.3 if it is not final, expand and add to the frontier
        new_nodes = expand(node, problem)
        for n in new_nodes:
            # check if it is expanded before adding to frontier
            if not np.any ([np.all(n["state"]== e) for e in expanded]):
             ##### CHANGE THE CONDITION TO FIT ALL SIZES ARRAYS::
             # if the state of N is not any of the elements in expanded, add to the frontier
            #if n["state"] not in expanded:
                frontier.append(n)

        # we compute the maximum size of frontier: the previous one or the current if it is bigger
        result["max_frontier"]=max(result["max_frontier"],len (frontier))
         # we compute the maximum depth: the previous one or the current if it is bigger
        result["max_depth"]=max(result["max_depth"], node["depth"])
         # we update the iterations count
        result["iterations"]= iterations

        iterations+=1
      #loop keeps running until no more nodes available or final state obtained

    result["final_state"] = node
    return(result)

"""##DFS"""

def DFS(problem):
    # result dictionary
    result = {"method":"DFS", "final_state":[], "status":"No nodes in the frontier. No solution possible.",
             "max_frontier":0, "max_depth":0, "iterations":0}

    # 1. problem definition
    initial_node={"state":problem.initial_state, "parent_node":{}, "actions":[], "cost":0, "depth":0, "evaluation":1}
    frontier = []

     # 2. add node to frontier
    frontier.append(initial_node)

    # 3. start exploring and expanding the frontier
    iterations=1
    while len (frontier)>0: #if we have elements in the frontier...
         # 3.1. get LAST element of frontier and delete it
        node = frontier[-1]
        frontier = frontier[:-1]

        # 3.2 check if it is final state:
        if problem.is_final_state (node["state"]):
            result["status"]="Solution Found."
            break #we end while. state will remain this last state computed, and sequence of actions will have all states.

        # 3.3 if it is not final, expand and add to the frontier
        new_nodes = expand(node, problem)
        for n in new_nodes:
            frontier.append(n)

        # we compute the maximum size of frontier: the previous one or the current if it is bigger
        result["max_frontier"]=max(result["max_frontier"],len (frontier))
         # we compute the maximum depth: the previous one or the current if it is bigger
        result["max_depth"]=max(result["max_depth"], node["depth"])
         # we update the iterations count
        result["iterations"]= iterations

        iterations+=1
      #loop keeps running until no more nodes available or final state obtained

    result["final_state"] = node
    return(result)

def DFS_g(problem):
    # result dictionary
    result = {"method":"DFS_g", "final_state":[], "status":"No nodes in the frontier. No solution possible.",
             "max_frontier":0, "max_depth":0, "iterations":0}

    # 1. problem definition
    initial_node={"state":problem.initial_state, "parent_node":{}, "actions":[], "cost":0, "depth":0, "evaluation":1}
    frontier = []

    ####
    expanded = []

     # 2. add node to frontier
    frontier.append(initial_node)

    # 3. start exploring and expanding the frontier
    iterations=1
    while len (frontier)>0: #if we have elements in the frontier...
         # 3.1. get first element of frontier and delete it
        node = frontier[-1]
        frontier = frontier[:-1]

        #add to expanded--> we add the state, as the rest of fields will be different
        expanded.append(node["state"])

        # 3.2 check if it is final state:
        if problem.is_final_state (node["state"]):
            result["status"]="Solution Found."
            break #we end while. state will remain this last state computed, and sequence of actions will have all states.

        # 3.3 if it is not final, expand and add to the frontier
        new_nodes = expand(node, problem)
        for n in new_nodes:
            # check if it is expanded before adding to frontier
            if not np.any ([np.all(n["state"]== e) for e in expanded]):
            # if n["state"] not in expanded:
                frontier.append(n)

        # we compute the maximum size of frontier: the previous one or the current if it is bigger
        result["max_frontier"]=max(result["max_frontier"],len (frontier))
         # we compute the maximum depth: the previous one or the current if it is bigger
        result["max_depth"]=max(result["max_depth"], node["depth"])
         # we update the iterations count
        result["iterations"]= iterations

        iterations+=1
      #loop keeps running until no more nodes available or final state obtained

    result["final_state"] = node
    return(result)

"""##IDS

"""

def IDS(problem, depth_limit, iteration_limit):
    # result dictionary
    result = {"method":"IDS"+str(depth_limit), "final_state":[], "status":"No nodes in the frontier. No solution possible.",
             "max_frontier":0, "max_depth":0, "iterations":0}

    # 1. problem definition
    initial_node={"state":problem.initial_state, "parent_node":{}, "actions":[], "cost":0, "depth":0, "evaluation":1}
    frontier = []

     # 2. add node to frontier
    frontier.append(initial_node)

    # 3. start exploring and expanding the frontier
    iterations=1

    # we add a control of the accepted max depth, starting with 1:
    current_max_depth=1

    #change the loop condition, now we have to control the end of the loop inside it
    while True:
        #if the len is 0, we have reached to the end of the depth. We start again with an increased max depth
        if len (frontier)==0:
            if current_max_depth<depth_limit:
                current_max_depth+=1
                node ={"state":problem.initial_state, "parent_node":{}, "actions":[], "cost":0, "depth":0, "evaluation":1}
                frontier.append(node)
            else:
                result["status"]= "No nodes in the frontier. No solution possible."
                break

        #control if we have reached the iteration limit, stop
        if iterations>iteration_limit:
            result["status"] = "Maximum number of iterations reached"
            break

         # 3.1. get LAST element of frontier and delete it
        node = frontier[-1]
        frontier = frontier[:-1]

        # 3.2 check if it is final state:
        if problem.is_final_state (node["state"]):
            result["status"]="Solution Found."
            break #we end while. state will remain this last state computed, and sequence of actions will have all states.

        # 3.3 if it is not final, expand and add to the frontier
        new_nodes = expand(node, problem)
        for n in new_nodes:
            ##### WE ONLY APPEND IF DEPTH <= CURRENT MAX DEPTH
            if n["depth"]<=current_max_depth:
                frontier.append(n)
            #in other case node is not appended, ending the loop

        # we compute the maximum size of frontier: the previous one or the current if it is bigger
        result["max_frontier"]=max(result["max_frontier"],len (frontier))
         # we compute the maximum depth: the previous one or the current if it is bigger
        result["max_depth"]=max(result["max_depth"], node["depth"])
         # we update the iterations count
        result["iterations"]= iterations

        iterations+=1
      #loop keeps running until no more nodes available or final state obtained

    result["final_state"] = node
    return(result)

def IDS_g(problem, depth_limit, iteration_limit):
    # result dictionary
    result = {"method":"IDS_g-"+str(depth_limit), "final_state":[], "status":"No nodes in the frontier. No solution possible.",
             "max_frontier":0, "max_depth":0, "iterations":0}

    # 1. problem definition
    initial_node={"state":problem.initial_state, "parent_node":{}, "actions":[], "cost":0, "depth":0, "evaluation":1}
    frontier = []

    ####
    expanded = []

     # 2. add node to frontier
    frontier.append(initial_node)

    # 3. start exploring and expanding the frontier
    iterations=1
     # we add a control of the accepted max depth, starting with 1:
    current_max_depth=1

    #change the loop condition, now we have to control the end of the loop inside it
    while True:
        #if the len is 0, we have reached to the end of the depth. We start again with an increased max depth
        if len (frontier)==0:
            if current_max_depth<depth_limit:
                current_max_depth+=1
                node ={"state":problem.initial_state, "parent_node":{}, "actions":[], "cost":0, "depth":0, "evaluation":1}
                frontier.append(node)
                expanded = []
            else:
                result["status"]= "No nodes in the frontier. No solution possible."
                break
       #before doing anything, if we have reached the iteration limit, stop
        if iterations>iteration_limit:
            result["status"] = "Maximum number of iterations reached"
            break
         # 3.1. get first element of frontier and delete it
        node = frontier[-1]
        frontier = frontier[:-1]

        #add to expanded--> we add the state, as the rest of fields will be different
        expanded.append(node["state"])

        # 3.2 check if it is final state:
        if problem.is_final_state (node["state"]):
            result["status"]="Solution Found."
            break #we end while. state will remain this last state computed, and sequence of actions will have all states.

        # 3.3 if it is not final, expand and add to the frontier
        new_nodes = expand(node, problem)
        for n in new_nodes:
            # check if it is expanded before adding to frontier
            if not np.any ([np.all(n["state"]== e) for e in expanded]):
            # if n["state"] not in expanded:
                if n["depth"]<=current_max_depth:
                    frontier.append(n)

        # we compute the maximum size of frontier: the previous one or the current if it is bigger
        result["max_frontier"]=max(result["max_frontier"],len (frontier))
         # we compute the maximum depth: the previous one or the current if it is bigger
        result["max_depth"]=max(result["max_depth"], node["depth"])
         # we update the iterations count
        result["iterations"]= iterations

        iterations+=1
      #loop keeps running until no more nodes available or final state obtained

    result["final_state"] = node
    return(result)

"""**MISCELANEOUS FUNCTIONS**"""

#First, we define two final variables to be used by these functions.
#these variables are the box (3x3 inner boxes) and grid (9x9 full board):

boxSide = 3
gridSide = boxSide * boxSide

"""getRandomlyFilledGrid() fills a 9x9 sudoku grid with numbers that aren't repeated in rows, columns or boxes.
Its output is alike the following:

[9, 3, 6, 8, 7, 5, 2, 1, 4]

[8, 5, 7, 4, 1, 2, 3, 6, 9]

[4, 2, 1, 9, 6, 3, 5, 7, 8]

[7, 9, 5, 1, 2, 8, 4, 3, 6]

[6, 4, 3, 7, 5, 9, 8, 2, 1]

[1, 8, 2, 6, 3, 4, 9, 5, 7]

[3, 1, 4, 5, 9, 6, 7, 8, 2]

[2, 7, 8, 3, 4, 1, 6, 9, 5]

[5, 6, 9, 2, 8, 7, 1, 4, 3]
"""

def getRandomlyFilledGrid():

    def pattern(r,c): return (boxSide*(r%boxSide)+r//boxSide+c)%gridSide

    # randomize rows, columns and numbers (of valid base pattern)
    def shuffle(s): return sample(s,len(s))
    rBase = range(boxSide)
    rows  = [ g*boxSide + r for g in shuffle(rBase) for r in shuffle(rBase) ]
    cols  = [ g*boxSide + c for g in shuffle(rBase) for c in shuffle(rBase) ]
    nums  = shuffle(range(1,boxSide*boxSide+1))

    # produce board using randomized baseline pattern
    board = [ [nums[pattern(r,c)] for c in cols] for r in rows ]

    return board

"""eraseUnecessaryNumbers() gets a board filled with numbers and it erases some of them
Output:

[0, 3, 0, 8, 0, 0, 0, 0, 0]

[0, 0, 7, 0, 0, 0, 0, 0, 0]

[4, 0, 0, 0, 0, 0, 0, 7, 0]

[7, 0, 5, 0, 0, 0, 4, 0, 6]

[0, 0, 0, 0, 0, 0, 0, 2, 0]

[0, 0, 0, 0, 3, 0, 9, 5, 0]

[0, 0, 0, 0, 0, 6, 0, 0, 0]

[0, 0, 8, 0, 4, 0, 0, 0, 5]

[0, 6, 0, 0, 8, 7, 0, 4, 0]

"""

def eraseUnecessaryNumbers(board):
    squares = gridSide*gridSide
    empties = squares * 3//4
    for p in sample(range(squares),empties):
        board[p//gridSide][p%gridSide] = 0

    numSize = len(str(gridSide))

"""createSudokuProblem() uses the previously defined functions and returns a sudoku problem."""

def createSudokuProblem():
    board = getRandomlyFilledGrid()
    eraseUnecessaryNumbers(board)

    return board

"""with printSudoku() instead of printing arrays we can print stuff that actually look like sudokus..."""

def printSudoku(board):

    if isinstance(board, pd.DataFrame): board = list(list(board.iloc[i]) for i in range(len(board)))

    def expandLine(line): return line[0]+line[5:9].join([line[1:5]*(boxSide-1)]*boxSide)+line[9:13]

    #types of lines
    line0  = expandLine("╔═══╤═══╦═══╗")
    line1  = expandLine("║ . │ . ║ . ║")
    line2  = expandLine("╟───┼───╫───╢")
    line3  = expandLine("╠═══╪═══╬═══╣")
    line4  = expandLine("╚═══╧═══╩═══╝")

    symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums   = [ [""]+[symbol[n] for n in row] for row in board ]

    print(line0)
    for r in range(1,gridSide+1):
        print( "".join(n+s for n,s in zip(nums[r-1],line1.split("."))) )
        print([line2,line3,line4][(r%gridSide==0)+(r%boxSide==0)])

"""This method splits an matrix into NxN smaller matrices"""

def split(board, nrows, ncols):
    board = board.to_numpy()
    r, h = board.shape
    return (board.reshape(h//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))

"""Gets the axis of the first blank space in the board"""

def getFirstSpace(state):

    state = state.to_numpy()
    index_zero = np.where(state == 0)
    return index_zero[0][0], index_zero[1][0]

"""this function returns the number that an action wants to write.

"""

def getNumber(action):
    return int(action[5])

"""# SUDOKU

First, mount the google drive unit to read the sudokus

"""

import pandas as pd
class problemSudoku ():
    #attributes of the class are empty
    name = ""
    boxSide = 3
    gridSide = boxSide * boxSide
    initial_state = {}
    goal_state = {}
    actions = []

    def __init__(self, sudoku : pd.DataFrame | None = None):
        
        sudoku1 = pd.read_csv("res/sample_sudokus/sudoku-1.txt", header=None)
        sudoku2 = pd.read_csv("res/sample_sudokus/sudoku-2.txt", header=None)
        sudoku3 = pd.read_csv("res/sample_sudokus/sudoku-3.txt", header=None)
        sudoku4 = pd.read_csv("res/sample_sudokus/sudoku-4.txt", header=None)

        #Assign name initial state, goal state, and possible actions

        if sudoku is None:
            self.initial_state = sudoku3
        else:
            self.initial_state = sudoku
        
        #self.goal_state =
        self.actions = [f'write{i}' for i in range(1,10)]


    def is_final_state(self, state):

        #We have reached the final state if there aren't any:
        # 1. blank spaces left
        #and no numbers are repeated in the same:
        # 2. row
        # 3. column
        # 4. box

        # 1. there can't be any blank spaces in the final state
        if not all(0 not in list(state[i]) for i in range(state.shape[1])): return False   #goes column by column

        #a set doesn't have repeated values. If this some row/col/box has repeated values, the size of the set will be less than 9
        # 2. rows
        for index, row in state.iterrows():

            if len(set(row)) < 9: return False

        # 3. cols
        for index in range(len(state)):

            if len(set(state[index])) < 9: return False

        # 4. boxes
        for box in split(state, boxSide, boxSide):

            if len(set(box.flatten())) < 9: return False

        #if none of the previous are given, then there aren't repeated values:
        return True

    def is_applicable (self, state, action):

        #checks if there are any blank spaces to be filled. If there are, return the row and column of the first one.
        if all(0 not in list(state[i]) for i in range(state.shape[1])): return False
        else: row, col = getFirstSpace(state)

        #get the Number that will we written.
        toWrite = getNumber(action)

        #If the number is not between 1 and 9, then we can't proceed. If it is, then check that the number isn't
        #already present in the row, column or box that will contain the new number.
        if toWrite not in range(1, 10): return False
        else:
            #rows
            if toWrite in list(state.iloc[row,:]):return False

            #cols
            if toWrite in list(state.iloc[:,col]): return False

            #box
            boxes = split(state, 3, 3)
            row1 = np.array([boxes[0], boxes[1], boxes[2]])
            row2 = np.array([boxes[3], boxes[4], boxes[5]])
            row3 = np.array([boxes[6], boxes[7], boxes[8]])
            boxes_matrix = np.array([row1, row2, row3])
            if toWrite in boxes_matrix[math.floor(row/3), math.floor(col/3)].flatten(): return False

            #If every test is passed, then the action is applicable.
            return True

    def effect (self, state, action):

        #get the row and column of the first space (or zero)
        row, col = getFirstSpace(state)

        #deep copy the actual state
        result = state.copy()

        #apply the given action
        result.iloc[row, col] = getNumber(action)

        #return the new state
        return result


# we also need cost of an action adn evaluation of the state, for other problems. for SUDOKU they are defined and return just 1
    def get_cost(self, action, state):
        return 1

    def get_evaluation (self, state):
        return 1

def main():
    p = problemSudoku()
    print("Initial state: ")
    printSudoku(p.initial_state)
    res = BFS(p)
    print("Solved sudoku: ")
    printSudoku(res["final_state"]["state"])

if __name__ == "__main__":
    main()

# p = problemSudoku()
# print (p.initial_state)
# res = DFS(p)
# print (res)

# p = problemSudoku()
# print (p.initial_state)
# res = BFS_g(p)
# print (res)

# p = problemSudoku()
# print (p.initial_state)
# res = DFS_g(p)
# print (res)

# p = problemSudoku()
# print (p.initial_state)
# res = IDS(p, 30000, 30000)
# print (res)

# p = problemSudoku()
# print (p.initial_state)
# res = IDS_g(p, 30000, 30000)
# print (res)