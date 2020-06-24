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

m, n = 12, 12
wordsGiven = ["Admissible", "Agent", "Backtrack", "Cannibal", "Deadend", "Global", "Graphsearch", "Heuristic", "Hill",
              "LISP", "Local", "Missionary", "Optimum", "Search", "Symmetry"]


def goal(state):
    grid = state[0]
    words = state[1]
    if len(words) == 0:
        return True
    else:
        return False


def precondition(rule, state):
    grid = state[0]
    words = state[1]
    word, row, col, dh, dv = rule
    for j in range(len(word)):
        if dh is 0 and dv is 0:
            return False
        if row >= len(grid):
            return False
        if col >= len(grid[row]):
            return False
        else:
            row += dv
            col += dh
    return True


# def precondition(rule, state):
#     grid = state[0]
#     words = state[1]
#     word, row, col, dh, dv = rule
#     # for j in range(len(word)):
#     if dh is 0 and dv is 0:
#         return False
#     hSpaceAvl = 0
#     if dh is 1:
#         # space available from right of [col] value till end of mat
#         hSpaceAvl = n - col
#     elif dh is -1:
#         # space available from beginning of mat till [col]
#         hSpaceAvl = col
#
#     # part 2: check if the word checks out the grid limit vertically
#     vSpaceAvl = 0
#     if dh is 1:
#         # space available from [row] value till end of mat
#         vSpaceAvl = m - row
#     elif dh is -1:
#         # space available from beginning of mat till [row] vertically
#         vSpaceAvl = row
#
#     if len(word) > hSpaceAvl or len(word) > vSpaceAvl:
#         return False
#     else:
#         return True

def applyRule(rule, state):
    grid = state[0]
    words = state[1]
    insertionWord = rule[0]
    row = rule[1]
    col = rule[2]
    dh = rule[3]
    dv = rule[4]

    i = 0
    while i < len(insertionWord):
        chari = insertionWord[i]
        grid[row][col] = chari
        row = row + dv
        col = col + dh
        i += 1

    index = words.index(insertionWord)
    words.pop(index)
    newState = (grid, words)
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


def flailWildly(state):
    grid = state[0]
    words = state[1]

    # from foxgoosecorn.py
    random.seed()  # use clock to randomize RNG
    numSteps = 0
    while not (goal(state)):
        numSteps += 1
        print("\n%d: ======\nstate=%s" % (numSteps, describeState(state)))
        rules = generateRules(state)
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


# mat = np.chararray((m, n))
mat = [["" for i in range(m)] for j in range(n)]

# rule = wordsGiven[7],0,0,1,0
# describeRule(rule)
#
# initialState = mat,wordsGiven
# describeState(initialState)
#
# state2 = applyRule(rule, initialState)
# describeState(state2)
#
# rule2=(wordsGiven[0],9,10,-1,-1)
#
# finalState = applyRule(rule2, state2)
# describeState(finalState)

initialState = mat, wordsGiven
flailWildly(initialState)
