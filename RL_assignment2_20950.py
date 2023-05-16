import numpy as np 
#RL environment
class gridworld:
    def __init__(self, dims, goal_states, error_states):
        # for actions
        self.actions = ["up", "down", "right", "left"]
        self.action_map = {"up": 0, "down":1, "right": 2, "left": 3}
        self.movement = [[-1,0], [1,0], [0,1], [0,-1]]	

        # for states
        self.rows = dims[0]
        self.columns = dims[1]
        self.current_state = None
        self.error_states = error_states 
        self.goal_states = goal_states
        self.empty_states =  [(i, j) for i in range(self.rows) for j in range(self.columns) if (i, j) not in (self.goal_states + self.error_states)]

        # visual representation of gridworld
        self.grid = np.array([["." for i in range(self.columns)] for j in range(self.rows)])
        for g in self.goal_states:
            self.grid[g] = "O"
        for e in self.error_states:
            self.grid[e] = "X"

    #print gridworld
    def print_grid(self):				
        print(self.grid)

    #set current state to start state
    def reset(self, start_state = None):
        if start_state != None:
             self.current_state = start_state
        else:
            self.current_state =  self.empty_states[np.random.choice(len(self.empty_states))]
        return self.current_state

    def step(self, action): 
        action_index = action
        action_taken = self.actions[action]  
        
        new_state, reward  = (-1, -1), 0 
        
        # out of the board , no movement but -reward for taking action
        if (action_taken == "up" and self.current_state[0] == 0) or (action_taken == "down" and self.current_state[0] == self.rows-1) :
            raise Exception("Invalid action taken")
        elif (action_taken == "left" and self.current_state[1] == 0) or (action_taken == "right" and self.current_state[1] == self.columns-1) :
            raise Exception("Invalid action taken")

        # inside the board
        else:
            new_state = (self.current_state[0] + self.movement[action_index][0] , self.current_state[1] + self.movement[action_index][1])
            if new_state in self.empty_states: # no reward in empty states 
                reward = -1
            elif new_state in self.goal_states: # good state
                reward = 10
            elif new_state in self.error_states: # bad state
                reward = -5 

        # current state changes because of taking action 
        self.current_state = new_state  

        #return current state, reward 
        return self.current_state, reward

class Solution:
    def __init__(self, env, gamma = 0.9, epsilon = 0.1, maxIter = 5000, maxTimesteps = 100):
        self.env = env
        self.q_table = np.zeros((self.env.rows, self.env.columns, len(self.env.actions)))
        self.gamma, self.epsilon, self.maxIter, self.maxTimesteps = gamma, epsilon, maxIter, maxTimesteps
    
    # Updates the visit_table for the calculation of alpha
    
    def update_visit(self, visit_table, state, action):
        visit_table[state[0], state[1], action] += 1

    # Choosing the action using epsilon greedy strategy
    # 1 - epsilon to choose greedy action, epsilon to choose random action
    # These actions are feasible 
        
    def epsilon_greedy(self, state):
        
        prob = np.random.random_sample() 
        feasible = self.FeasibleActions(state)
        if prob > self.epsilon:
            # greedy
            action = feasible[np.argmax(self.q_table[state[0], state[1], feasible])]
        else:
            # random
            action = np.random.choice(feasible)
        return action
    
    # Making the entries of q_table to -inf for all the infeasible actions at any particular state
    # Try and Expect checks for an exception and catches the actions which are non feasible in Infeasible_actions
    
    def MakeInfeasibleInf(self):
        for i in range(self.env.rows):
            for j in range(self.env.columns):
                self.env.current_state = (i, j)
                Infeasible_actions = []
                for k in range(len(self.env.actions)):
                    try:
                        self.env.step(k)
                    except:
                        Infeasible_actions.append(k)
                        continue
                    self.env.current_state = (i,j)
                for k in Infeasible_actions:
                    self.q_table[i, j, k] = -np.inf
    
    # Finds the feasible actions at a given state
    
    def FeasibleActions(self, state):
        actions = []
        self.env.current_state = state
        for action in self.env.actions:
            actions.append(self.env.action_map[action])
        for i in range(len(self.env.actions)):
            try:
                self.env.step(i)
            except:
                actions.remove(i)
                continue
            self.env.current_state = state
        return actions
    
    # Define the Q-learning algorithm
    def q_learning(self):
        
        # Make entires in q_table -inf for all infeasible actions at every state  
       
        self.MakeInfeasibleInf()

        for iter in range(self.maxIter):
            
            state = self.env.reset()
            visit_table = np.zeros((self.env.rows, self.env.columns, len(self.env.actions)))
            
            for steps in range(self.maxTimesteps):
            
                # epsilon - greedy
                # choose action
                # Only feasible actions are returned
                
                current_action = self.epsilon_greedy(state)
                
                next_state, Reward = self.env.step(current_action)
                self.update_visit(visit_table, state, current_action)

                alpha = 1 / visit_table[state[0], state[1], current_action]

                # Q update 

                self.q_table[state[0], state[1], current_action] += alpha * (Reward + self.gamma * np.max(self.q_table[next_state[0],next_state[1]]) - self.q_table[state[0], state[1], current_action])
                state = next_state

        
        return self.q_table 


if __name__ == "__main__":
    
    np.random.seed(100)
    size = 3
    goal_states = [(0, size-1), (1, size-1), (size-1, 0)]
    error_states = [(o, o) for o in range(size)] 
    grid = gridworld((size, size), goal_states, error_states)

    grid.print_grid()
    solution = Solution(grid)
    q_table = solution.q_learning()

    print("\nQ-table:\n")
    print(q_table)
    assert round(q_table[0,0,1], 5) == 75.5 and round(q_table[0,1,2], 5) == 100.0, 'wrong answer'