"""

Word Puzzle Generator:

Author: Shivansh Suhane, ss4328
April 15, 2020

Program description:

Knowledge base: MxN grid, list of k words to be placed in the gru
State: s = (grid,words) -> partally filled in grid and a list of words that need to be added


Rule: (word, row, col, dh, dv)
Action: Insert word in grid starting at pos [row,col] and continuing in [dh,hv] direction
     |dh| + |dv| > 0
    -1<=dh, dv<=1      meaning-> dh dv can have values -1,0,1
     Preconditions: Word doesn't extend beyond the edge of grid, word doesn't conflict with existing words in grid
"""

import numpy as np
import math
import itertools
import random
import collections

occupiedLocs = {}
m, n = 11, 11
puzzleAttempt = 1
# wordsGiven = ["Admissible", "Agent", "Backtrack", "Cannibal", "Deadend", "Global", "Graphsearch", "Heuristic", "Hill",
#               "LISP", "Local", "Missionary", "Optimum", "Search", "Symmetry"]
wordsGiven = ["admissible", "agent", "backtrack", "cannibal", "deadend", "global", "graphsearch", "heuristic", "hill",
              "lISP", "local", "missionary", "optimum", "search", "symmetry"]
depthBound = 15

def goal(state):
    grid = state[0]
    words = state[1]
    if len(words) == 0:
        return True
    else:
        return False

'''

Precondition: checks the rule for application validity 

Returns false for: 
1. if hd,hv both are zero
2. if location+direction combo doesn't have sufficient space for the imminent insertion
3. if the imminent insertion would replace chars from any legal insertions previously applied

Returns true if (1), (2), and (3) are not met
'''
def precondition(rule, state):
    grid = state[0].copy()
    words = state[1].copy()

    word, row, col, dh, dv = rule.copy()
    # for j in range(len(word)):
    if (dh == 0 and dv==0):
        return False
    hSpaceAvl = 0
    vSpaceAvl = 0


    if dh is 1:
        # space available from right of [col] value till end of mat
        hSpaceAvl = n - col
    elif dh is -1:
        # space available from beginning of mat till [col]
        hSpaceAvl = col + 1
    else:
        hSpaceAvl = len(word)       #so that the spacing conditions are met, we ignore the hspaceavl condition




    # part 2: check if the word checks out the grid limit vertically

    if dv is 1:
        # space available from [row] value till end of mat
        vSpaceAvl = m - row
    elif dv is -1:
        # space available from beginning of mat till [row] vertically
        vSpaceAvl = row + 1
    else:
        vSpaceAvl = len(word)   #so that the spacing conditions are met, we ignore the vspaceavl condition


    # check if we have space available for the word about to be inserted
    if len(word) > hSpaceAvl or len(word) > vSpaceAvl:
        return False
    else:
        # check if the word being inserted will wrongly replace any existing character while its insertion
        if(wronglyReplacesExisting(rule, state)):
            return False

    return True

'''
Helper function to assist Precondition; returns true if any of the word's char tries to override a legally inserted word previously
'''
def wronglyReplacesExisting(rule, state):
    grid = state[0]
    words = state[1]
    word, row, col, dh, dv = rule

    # loop over the word itself to get chars
    for i in range(0, len(word)):
        loc = row, col
        chari = word[i]

        # if location is in occupiedLoc dictionary
        if(loc in occupiedLocs):
            # if wrongly replaces existing char
            currentChar = occupiedLocs[loc]         # get the char currently in the location
            if(chari!=currentChar and currentChar != " "):      # if contender char is di
                return True                         # if the currently occupied char is different than the char trying to be inserted, return false

        row = row + dv
        col = col + dh
        i += 1

    return False


def applyRule(rule, state):
    grid = state[0].copy()
    words = state[1].copy()
    insertionWord = rule[0]
    row = rule[1]
    col = rule[2]
    dh = rule[3]
    dv = rule[4]


    newGrid = grid.copy()
    newWordsList = words.copy()

    i = 0
    while i < len(insertionWord):
        chari = insertionWord[i]
        # print("inserting: ", insertionWord)
        # describeState(state)
        # describeRule(rule)
        newGrid[row][col] = chari
        occupiedLocs[(row,col)] = chari
        row = row + dv
        col = col + dh
        i += 1

    index = words.index(insertionWord)
    newWordsList.pop(index)
    newState = (newGrid, newWordsList)
    return newState


def describeState(state):
    print("Currently the word matrix is:")
    # state[0].decode("utf-8")
    print(np.matrix(state[0]))
    print("Words remaining to be entered:")
    print(state[1])


def describeRule(rule):
    return ("Place the word " + str(rule[0]) + " in the grid starting at position (" + str(rule[1]) + ", " + str(
        rule[2]) + ") and proceeding in the direction [" + str(rule[3]) + "," + str(rule[4]) + "].")

'''
Calls precondition and returns a list of all possible rules that may be applied to the current state
This should be a list of all possible words, starting positions and directions which satisfy the preconditions for the given state
'''
def generateRules(state):
    grid = state[0]
    words = state[1]
    rules = []
    for x, row in itertools.product(range(len(words)), range(len(grid))):
        for dh, dv in itertools.product([-1, 1, 0], [-1, 1, 0]):
            for col in range(len(grid[row])):
                rule = (wordsGiven[x], row, col, dh, dv)
                if 0 < row < m and 0 < col < n:
                    if precondition(rule, state):
                        rules.append(rule)
    return rules


def generateRules2(state):
    grid = state[0]
    words = state[1]

    rules = []
    for wordIndex in range(0,len(words)):
        for row in range(0,m):
            for col in range(0,n):
                for dh in range(-1,2):
                    for dv in range(-1,2):
                        rule = [words[wordIndex], row, col, dh, dv]
                        if (precondition(rule, state)):
                            rules.append(rule)

    # SPEED UPDATE: makes rules using numpy; therefore faster, but needs rule structure to be changed from words[i] to i simply
    # shape = (8 - 0, 12 - 0, 12 - 0, 2 - (-1), 2 - (-1))
    # rules = np.zeros(shape)
    # # create array of indices
    # rules = np.argwhere(rules == 0).reshape(*shape, len(shape))
    # # correct the ranges that does not start from 0, here 4th and 5th elements (dh and dv) reduced by -1 (starting range).
    # # You can adjust this for any other ranges and elements easily.
    # rules[:, :, :, :, :, 3:5] -= 1

    # print(rules[:5])
    return rules

def flailWildly(state):
    grid = state[0]
    words = state[1]

    # from foxgoosecorn.py
    random.seed()  # use clock to randomize RNG
    numSteps = 0
    while not (goal(state)):
        numSteps += 1
        print("\n%d: ======\nstate=%s" % (numSteps, describeState(state)))
        rules = generateRules2(state)
        print("There are %d applicable rules" % (len(rules)))
        for i in range(len(rules)):
            print(str(i) + ": -- " + describeRule(rules[i]))
        if len(rules) is 0:
            print("No rules available anymore")
            break
        r = random.randint(0, len(rules) - 1)
        print("Choosing rule[%s]=%s" % (r, describeRule(rules[r])))
        state = applyRule(rules[r], state)

    print("Stopped at state %s" % (describeState(state)))

'''
Basically an algorithm to find a solution that brute forces flailWildly till a solution is found by continuously re-initiating at different states on every run
'''
def flailWildlyForSuccess():
    global puzzleAttempt

    state = initializeState()
    grid = state[0]
    words = state[1].copy()

    state = sortWordsByLength(state)

    # from foxgoosecorn.py
    random.seed()  # use clock to randomize RNG
    numSteps = 0

    while not (goal(state)):
        numSteps += 1
        print("\n%d: ======\nstate=%s" % (numSteps, describeState(state)))
        rules = generateRules2(state)
        print("There are %d applicable rules" % (len(rules)))
        for i in range(len(rules)):
            print(str(i) + ": -- " + describeRule(rules[i]))
        if len(rules) is 0:
            print("No rules available anymore")
            break
        r = random.randint(0, len(rules) - 1)
        print("Choosing rule[%s]=%s" % (r, describeRule(rules[r])))
        state = applyRule(rules[r], state)

    puzzleAttempt = puzzleAttempt + 1
    SUCCESS='INCOMPLETE PUZZLE'
    if(goal(state)):
        SUCCESS = "COMPLETE PUZZLE"

    print("Stopped at state: ", SUCCESS)

    print("Retrying Puzzle. Atempt# ", puzzleAttempt)
    if goal(state)!=True:
        flailWildlyForSuccess()



'''
Basically resets the puzzle by generating new grid, reinitializing words list (it resets somehow) and resetting occupiedLocs to an empty dictionary
'''
def initializeState():
    global occupiedLocs
    mat = [[" " for i in range(m)] for j in range(n)]
    # mat = np.asarray(mat)
    occupiedLocs = {}

    newState = mat.copy(), wordsGiven.copy()
    return list(newState)

# def getCompletionStatus(state, applicableRules):
#     grid = state[0]
#     words = state[1]
#
#     if(len(words)!=0 && len(applicableRules)):


def sortWordsByLength(state):
    words = state[1].copy()

    words.sort()
    words.sort(key=len, reverse=True)
    return state[0], words

def isDeadend(state):           #number of chars to insert>number of positions
    charCount = 0
    gridCapacity = m*n

    for word in state[1]:
        for i in word:
            charCount+=1

    if(charCount>gridCapacity):
        return True

    return False


def inRestOfList(state, stateList):

    if(stateList.count(state)>1):
        return True
    else:
        return False
 
'''
Basically an algorithm to find a solution that abandons each partial rule if the rule cannot possibly yield a valid solution
returns a list of rules from start to goal

Base cases: cycle                       StateList contains the state being tested
            deadend                     0 applicable rules, words still left ??
            goal                        ANS
            len(state)>depthbound       To prevent infinte looping
            ruleset is null             No moves to apply!
            end of loop after recur     Nothing worked so STOP
            
Recursive case: Attach path to path list

'''
def backtrack(stateList):
    state = stateList[0]                    #current State

    if(inRestOfList(state,stateList)):                  #cycle
        return 'FAILED-1'
    if(isDeadend(state)):
        return 'FAILED-2'
    if(goal(state)):
        return None
    if(len(stateList)>depthBound):
        return 'FAILED-3'

    ruleSet = generateRules2(state)

    if(ruleSet==None):
        return 'FAILED-4'

    for r in ruleSet:
        describeRule(r)
        # visitedStates = [state] + visitedStates
        newState = applyRule(r,state)
        describeState(newState)
        newStateList = [newState] + stateList
        # newStateList = stateList.insert(0,newState)
        path = backtrack(newStateList)
        if path != "FAILED-1" and path != "FAILED-2" and path != "FAILED-3" and path != "FAILED-4" and path != "FAILED-5":
            return path.append(r)

    return 'FAILED-5'


if __name__ == '__main__':

    #debugging 1
    # rule = wordsGiven[7],0,0,1,0
    # describeRule(rule)
    # describeState(initialState)
    #
    # state2 = applyRule(rule, initialState)
    # describeState(state2)
    #
    # rule2=(wordsGiven[0],9,10,-1,-1)
    #
    # if(precondition(rule2, state2)):
    #     finalState = applyRule(rule2, state2)
    #     describeState(finalState)


    #debugging 2: checking generateRules


    # allRules1 = []
    # allRules1 = generateRules(initialState)
    # allRules1 = np.asarray(allRules1)
    #
    # print(allRules1)
    # print(allRules1.shape)



    #uptimate debug test
    # flailWildly(initialState)

   # flailWildlyForSuccess()     # doesn't work. WHY???
    stateInitial = initializeState()

    rules1 = generateRules2(stateInitial)

    # state1 = applyRule()
    # visitedStates = []
    stateList = [stateInitial]
    backtrack(stateList)
