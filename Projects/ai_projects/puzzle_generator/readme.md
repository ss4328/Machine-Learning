# Puzzle Generator

## Problem Statement
This project is focused on one problem: Gernerating a M x N puzzle provided a set of words. I'll create an artificially-intelligent system to solve this problem. The problem is described here;
https://www.armoredpenguin.com/wordsearch/

## PEAS Analysis
PEAS stands for Performance measure, Environment, Actuators, Sensors.
Performance Measure: Words inserted, failure count
Enfironment: State - MxN grid, and words list
Actuator: Function called solve()
Sensors: Goal() Function

## Algorithm & Working
This is a fairly large problem, I'll try to break it in parts to solve it efficiently. 
- Part 1 will generate the grid, set up environment
- Part 2 will solve using uninformed search - backtracking

### V1.0 - Setting up the Environment
- Assign an empty m×n grid and the full list of words {word[0], ..., word[k-1]} to initialState.
- Write a function goal(state) which returns true if state.words equals the empty list.
- Write a function applyRule(rule,state) which returns the value of applying a rule to a given state. This does not change the value of state.
- Write a function precondition(rule,state) which returns True if the given rule may be applied to state, that is, the preconditions are satisfied. For instance, given the value of initialState, the precondition for rule (7,0,0,1,0) is satisfied (and True is returned) because the word fits in the grid and there are no other words in the grid that conflict with it. Note, however that False is returned for the rule (7,0,8,1,0), because the word does not fit in the grid when starting at position [0,8] and filling forward in the positive direction. Likewise, suppose "heuristic" has been placed successfully with the first rule and the new state is state2. Then if rule2 is (0,9,10,-1,-1) , precondition(rule2,state2) returns True because the final letter 'e' overlaps with the letter 'e' in position [0,1]. However, the rule (0,9,11,-1,-1) would not satisfy the preconditions for this state because it does not overlap properly with existing words in the grid.
- Write a function generateRules(state) which calls precondition and returns a list of all possible rules that may be applied to the current state, where rules have the form described above. This should be a list of all possible words, starting positions and directions which satisfy the preconditions for the given state.
Write a function describeState(state), which shows the partially filled in grid and lists words still remaining to be placed in the grid.
- Write a function describeRule(rule), which explains the meaning of the given rule, e.g.,
Place the word "admissible" in the grid starting at position (9,10) and proceeding in the direction [-1,-1].
- Test these primitives by writing a routine flailWildly(state), which repeatedly tests goal(state) and if a goal has not been reached, determines all applicable rules for the current state, chooses one randomly and applies it. At each step, describe the current state, describe each applicable rule, and describe which rule has been chosen.

### V2.0 - Uninformed Search
Instead of using a random, exhaustive filler, use backtracking to finish the puzzle. 
- Use the describeState and describeRule functions to report which rules are tried and which states are reached. It's beneficial to print occasional blank or dotted lines, to enhance readability. Output should also include the solution path and a report on the number of calls to backTrack, and number of failures before finding the solution. Important note: Make certain that your applyRule(rule,state) function does not change the original state, but returns a new state. This is particularly important when using a tentative strategy like backTrack.

## Result
Backtracking solution works! After 300 failures, in a search tree of about 3 milion, the program gives out a correct result!


