import numpy as np
import pandas as pd
import pprint
import random
import time

#Some constant variables
dice1 = {0,1}
dice2 = {0,1,2}
actions = ['d1', 'd2', 'EXIT']
dice_int = {'d1':1, 'd2':2} #it will be used to do the strategy's conversion
dice_string = {1:'d1', 2:'d2'} #same
gamma = 0.9
epsilon = 0.000000001
other_choice = {1:2, 2:1} #used to randomize a strat

#those variables will be assigned in the 'init' function
states = None 
reward = None 

#simple examples of strat 
strat_1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
strat_2 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

#example of board :
layout_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#layouts for the tests
layouts = [
[0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 4, 4, 0],
[0, 1, 0, 4, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 4, 0, 3, 0, 3, 0, 0, 2, 0],
[4, 3, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 1, 3, 4, 1, 0, 0, 0, 0, 0, 4, 0, 0, 3, 0],
[1, 3, 4, 0, 4, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0],
[0, 0, 4, 0, 3, 1, 4, 2, 2, 0, 0, 0, 0, 3, 0],
[0, 0, 3, 1, 1, 2, 1, 3, 0, 4, 0, 0, 0, 0, 0],
[2, 1, 3, 0, 3, 0, 4, 0, 3, 1, 0, 0, 0, 3, 0],
[0, 4, 4, 1, 1, 0, 1, 3, 2, 0, 0, 0, 3, 0, 0],
[0, 2, 0, 2, 3, 0, 2, 3, 1, 1, 3, 1, 0, 0, 0],
[0, 1, 3, 0, 3, 2, 4, 1, 2, 4, 4, 0, 0, 0, 0],
[4, 1, 3, 2, 2, 2, 3, 0, 2, 0, 3, 0, 1, 0, 0],
[2, 2, 0, 4, 4, 4, 3, 4, 3, 0, 0, 3, 4, 0, 0]]

############### functions related to the Markov Decision Process ###############
def init(n):
    '''
        param n : number of states + number of traps type 3 and 4
        do : initialize 'reward' and 'states' depending on n
    '''
    global states
    states = range(1,16+n)
    global reward
    reward = {1 : 1,
        2 : 1,
        3 : 1,
        4 : 1,
        5 : 1,
        6 : 1,
        7 : 1,
        8 : 1,
        9 : 1,
        10 : 1,
        11 : 1,
        12 : 1,
        13 : 1,
        14 : 1,
        15 : 0}

    for i in range(1, n+1):
        reward[15+i] = 1

def convert(dict, dice):
    '''
        param dict : contains the results of value iteration (expected cost or dice, see param dice)
        param dice : boolean that tells if we are convering the 'Dice' of the 'Expec' array
        return : dict in the requested format (as told in the project guidelines)
    '''
    temp = np.array(list(dict.values()))
    if dice:
        temp_bis = [0] * 14
        for i in range(0,14):
            temp_bis[i] = dice_int.get(temp[i])
        return np.array(temp_bis)

    return temp[0:14]

def make_transition(board, circle=False):
    """
        param layout: type <numpy.ndarray> (dim 1x15) containing
                        0 for ordinary square
                        1 for restart trap
                        2 for penalty trap
                        3 for prison trap
                        4 for mystery trap
        return: dictionary {state: {action : [(probability, new state) ] } }
    """

    def add_trans(dice, proba, state):
        """
            param dice: type <list> of tuples (probability, state); list of states and their associated probability
            param proba: type <float> probability
            param state: type <int> state
            return: dice with (proba,state) added. If state already in dice, the old and the new proba are summed up
        """
        done = False
        for i in range(len(dice)):
            if dice[i][1] == state:
                el = dice.pop(i)
                #print(el)
                dice.append((el[0] + proba, state))
                done = True
                break

        if not done:
            dice.append((proba,state))
        return dice

    def add_trap(dice, new_state, layout, proba=1/3, trap = 0):
        """
            param dice: type <list> of tuples (probability, state); list of states and their associated probability
            param new_state: type <int> state
            param layout: type <numpy.ndarray> (dim 1x15) containing
                        0 for ordinary square
                        1 for restart trap
                        2 for penalty trap
                        3 for prison trap
                        4 for mystery trap
            param proba: type <float> probability
            return: same as add_trans but with traps enabled
        """
        # ordinary square
        if layout[new_state-1] == 0 and trap == 0:
            return add_trans(dice, proba, new_state)
        # restart trap
        elif layout[new_state-1] == 1 or trap == 1:
            return add_trans(dice, proba, 1)
        # penalty trap
        elif layout[new_state-1] == 2 or trap == 2:
            if new_state-3 < 1 :
                return add_trans(dice, proba, 1)
            elif new_state == 11:
                return add_trans(dice, proba, 1)
            elif new_state == 12:
                return add_trans(dice, proba, 2)
            elif new_state == 13:
                return add_trans(dice, proba, 3)
            else:
                return add_trans(dice, proba, new_state-3)
        # prison trap
        elif layout[new_state-1] == 3 or trap == 3:
            prison_state = 16 + prisons.index(new_state)
            return add_trans(dice, proba, prison_state)
        # mystery trap
        elif layout[new_state-1] == 4 and trap == 0:
            d1 = add_trap(dice, new_state, layout, proba/3, 1)
            d2 = add_trap(d1, new_state, layout, proba/3, 2)
            return add_trap(d2, new_state, layout, proba/3, 3)
            pass

    transition = {}

    indices = [i for i, x in enumerate(board) if x == 3 or x == 4]
    prisons = [x + 1 for x in indices]
    layout = board.copy()
    for prison in prisons:
        layout.append(0)



    for s in range(1,16):
        if s == 15:
            transition[s] = {'EXIT' : [(0.0, s)]}
            break

        actions = {}

        # dice n°1
        d1 = []
        for jet in dice1:

            if s + jet > 15:
                if circle:
                    d1 = add_trans(d1, 0.5, 1)
                else:
                    d1 = add_trans(d1, 0.5, 15)
            elif s == 10:
                if jet == 0:
                    d1 = add_trans(d1, 0.5, s + jet)
                else:
                    d1 = add_trans(d1, 0.5, 15)
            elif s == 3:
                if jet == 0:
                    d1 = add_trans(d1, 0.5, s + jet)
                if jet > 0:
                    d1 = add_trans(d1, 0.25, s + jet)
                    d1 = add_trans(d1, 0.25, 10 + jet)

            else:
                d1 = add_trans(d1, 0.5, s + jet)

        actions['d1'] = d1



        # dice n°2
        d2 = []
        for jet in dice2:

            if s + jet > 15:
                if circle:
                    d2 = add_trap(d2, 1, layout)
                else:
                    d2 = add_trans(d2, 1/3, 15)

            elif s == 9:
                if jet == 0 or jet == 1:
                    d2 = add_trap(d2, s + jet, layout)
                else:
                    d2 = add_trans(d2, 1/3, 15)
            elif s == 10:
                if jet == 0:
                    d2 = add_trap(d2, s + jet, layout)
                elif jet == 1:
                    d2 = add_trans(d2, 1/3, 15)
                else:
                    if circle:
                        d2 = add_trap(d2, 1, layout)
                    else:
                        d2 = add_trans(d2, 1/3, 15)

            elif s == 3:
                if jet == 0:
                    d2 = add_trap(d2, s + jet, layout)
                if jet > 0:
                    d2 = add_trap(d2, s + jet, layout, 1/6)
                    d2 = add_trap(d2, 10 + jet, layout, 1/6)
            else:
                d2 = add_trap(d2, s + jet, layout)

        actions['d2'] = d2
        transition[s] = actions



    # Handle prison states
    for i in range(len(prisons)):
        s = 16 + i
        p = prisons[i]

        actions = {}

        # dice n°1
        d1 = []
        for jet in dice1:
            if p == 10:
                if jet == 0:
                    d1 = add_trans(d1, 0.5, 10)
                else:
                    d1 = add_trans(d1, 0.5, 15)
            elif p == 3:
                if jet == 0:
                    d1 = add_trans(d1, 0.5, 3)
                elif jet > 0:
                    d1 = add_trans(d1, 0.25, 4)
                    d1 = add_trans(d1, 0.25, 11)
            else:
                   d1 = add_trans(d1, 0.5, p + jet)

        actions['d1'] = d1



        # dice n°2
        d2 = []
        for jet in dice2:

            if p + jet > 15:
                if circle:
                    d2 = add_trap(d2, 1, layout)
                else:
                    d2 = add_trans(d2, 1/3, 15)
            elif p == 9:
                if jet == 0 or jet == 1:
                    d2 = add_trap(d2, p + jet, layout)
                else:
                    d2 = add_trans(d2, 1/3, 15)
            elif p == 10:
                if jet == 0:
                    d2 = add_trap(d2, 10, layout)
                elif jet == 1:
                    d2 = add_trans(d2, 1/3, 15)
                else:
                    if circle:
                        d2 = add_trap(d2, 1, layout)
                    else:
                        d2 = add_trans(d2, 1/3, 15)

            elif p == 3:
                if jet == 0:
                    d2 = add_trap(d2, 3, layout)
                if jet > 0:
                    d2 = add_trap(d2, p + jet, layout, 1/6)
                    d2 = add_trap(d2, 10 + jet, layout, 1/6)
            else:
                d2 = add_trap(d2, p + jet, layout)

        actions['d2'] = d2
        transition[s] = actions


    return transition

def value_iteration(transition, reward):
    '''
        param transition : corresponds to the transition matrix as a dictionnary
        param reward : correspond to the reward (cost actually) of each square
        return : V containing the results of value iteration
    '''

    T = transition
    R = reward

    V1 = {s: 0 for s in states}
    while True:

        V = V1.copy()
        delta = 0
        for s in states:
            V1[s] = min([R[s] + sum([p *V[s1] for (p, s1) in T[s][a]]) for a in actions if a in T[s].keys()])
            delta = max(delta, abs(V1[s] - V[s]))

        if delta < 0.00000000000000000001:
            return V

def dice_choice(V, T):
    '''
        param V : resuts of value iteration
        param T : transition matrix
        return : Dice containing the strategy resulting of V and T
    '''
    Dice = {}
    for s in states:
        Dice[s] = min(T[s].keys(), key=lambda a: expected_cost(a, s, V, T))
    return Dice

def expected_cost(a, s, V, T):
    return sum([p * V[s1] for (p, s1) in T[s][a]])

def markovDecision(layout, circle):
    '''
        param layout : contains the type of square for each square
        param circle : bool, tells if the bordle is circle or not
        return : the optimal strategy for the game problem
    '''
    transition = make_transition(layout, circle)

    init(len(transition)-15)

    Expec  = value_iteration(transition, reward)

    Dice = dice_choice(Expec, transition)

    return [convert(Expec,False), convert(Dice, True)]

############### functions related to the Game Simulation #######################

def potential_next_squares(transition, square, choice):
    '''
        param transition : transition matrix as dictionnary
        param square : current_square of the player
        param choice : dice choice of the player
        return : tuple containing all the possible next_squares with their probas based on the param
    '''
    tuples_squares_proba = transition[square][choice]
    squares = [None] * len(tuples_squares_proba)
    probas = [None] * len(tuples_squares_proba)
    for i in range(0, len(tuples_squares_proba)):
        squares[i] = tuples_squares_proba[i][1]
        probas[i] = tuples_squares_proba[i][0]

    return [squares, probas]

def game(layout, circle, strat):
    '''
        param layout : represents the board
        param circle : bool, tells if the board is circle or not
        param strat : a dice strategy
        return : the empirical cost of a strategy based on the board (layout and circle)

        this function is used to simulate the game and compute the empirical cost
    '''
    #FIRST, we check the type(strat) : if int => convert, else don't
    if not isinstance(strat[0], str):
        strat = strat_convert(strat)

    #NOW we are ready to launch the game
    transition = make_transition(layout, circle)
    cost = 0
    curr_square = 1
    running = True

    while running:
        choice = strat[(curr_square % 15)]

        #select the possible next_squares and their probas from the transition matrix
        #as a tuple of lists
        potential = potential_next_squares(transition, curr_square, choice)

        next_square = random.choices(potential[0], potential[1])

        curr_square = next_square[0]
        cost = cost + 1

        if curr_square == 15:
            running=False

    return cost

def strat_convert(strat):
    '''
        param strat : a strategy in a 'int' format [1,1,2,1,2,1,...]
        return new_strat : in a 'string' format ['d1','d1','d1','d2']
    '''
    new_strat = [None] * len(strat)
    for i in range(0,len(strat)):
        new_strat[i] = dice_string.get(strat[i])

    return new_strat

############## functions related to the comparison of strategies ###############

def run_experiments(strats, layouts, n_games, circle):
    '''
        param strats : a list of strategy to test. Also 'markov' and 'random_markov'
        param layouts : a list of layout on which we apply the game
        param n_games : number of iterations 
        param circle : bool, tells if circle or not
        return results : a list of Pandas Dataframe, each corresponding to a layout and 
                        containing the resuts of the n_games iteration for all strats

    This function do a comparison of different strategies on different layouts and 
    report in pandas dataframes. The function will do n_games iteration for each strategy
    for each layout.
    if a strat is 'markov', then compute the optimal markov strat for the specific layout
    if a strat is 'random_markov', then compute the randomized optimal markov strat
    '''
    #create a tab of size len(layouts), each item containing a dataframe
    results = [None] * len(layouts)

    #run expe for each layout
    for r in range(0,len(results)):
        print('layout' + str(r) + ' is running')
        res = pd.DataFrame(index=np.arange(n_games), columns=np.arange(len(strats)))
        #run expe for each strats
        for i in range(0,len(strats)):
            #run expe n_games times
            if strats[i] == 'markov':
                #we append a one-value list to the strat because our 'game' function deal with strat of length 15
                strat_temp = np.append(markovDecision(layouts[r], circle)[1], [1])
            elif strats[i] == 'random_markov':
                strat_temp = randomize_strat(np.append(markovDecision(layouts[r], circle)[1], [1]))
            else:
                strat_temp = strats[i]
            for j in range(0,n_games):
                res.iloc[j,i] = game(layouts[r], circle, strat_temp)
        results[r] = res

    return results

def randomize_strat(strat):
    '''
        param strat : a strategy to randomize
        return : a new strategy following that rule : p=0.8 to do the same choice, p=0.2 to make the other choice
    '''
    new_strat = [None] * len(strat)

    for i in range(0,len(strat)):
        new_strat[i] = random.choices([strat[i], other_choice[strat[i]]], [0.8, 0.2])[0]
    
    return np.array(new_strat)

def mean_experiments(results):
    '''
        param results : a unique dataframe containing the results of differents strats on a layout
                        as described in the 'run_experiments' function
        return : means, a list containing the mean empirical cost for each strategy
    '''
    means = [None] * len(results.columns)
    for i in range(0, len(means)):
        means[i] = results[i].mean()

    return means

def generate_random_strat():
    '''
        return : a randomly generated strategy
    '''
    strat = [None] * 15
    for i in range(0,15):
        strat[i] = random.choices([1,2], [0.5,0.5])[0]

    return strat
    
################################################################################

####### Run this part to recompute the results presentend in the report  #######


results_circle = run_experiments([strat_1, strat_2, generate_random_strat(), 'markov', 'random_markov'],
    layouts, 1000, True) 

results_no_circle = run_experiments([strat_1, strat_2, generate_random_strat(), 'markov', 'random_markov'],
    layouts, 1000, False) 

#Uncomment this part to update the experiments files
''' 
for i in range(0,len(results_circle)):
    results_circle[i].to_pickle('experiments/circle/layout_' + str(i)+'_expe.pkl')

for i in range(0,len(results_no_circle)):
    results_no_circle[i].to_pickle('experiments/no_circle/layout_' + str(i)+'_expe.pkl')
'''



