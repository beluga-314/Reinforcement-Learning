import numpy as np
import math


class Maze:
    def __init__(self, gridHeight=6, gridWidth=6, terminalReward=10, lockPickProb=0.5):
        self.rewardsLeft = np.array([[-1, 0, 0, 0, 0, 0],
                                     [-1, -1, 0, 0, 0, -10],
                                     [-1, 0, 0, -1, -1, -1],
                                     [0, 0, 0, -10, -1, -1],
                                     [-1, -1, 0, 0, -1, 0],
                                     [-1, 0, -1, 0, 0, -1]])

        self.rewardsRight = np.array([[0, 0, 0, 0, 0, -1],
                                      [-1, 0, 0, 0, -10, -1],
                                      [0, 0, -1, -1, -1, -1],
                                      [0, 0, -10, -1, -1, -1],
                                      [-1, 0, 0, -1, 0, -1],
                                      [0, -1, 0, 0, -1, -1]])

        self.rewardsUp = np.array([[-1, -1, -1, -1, -1, -1],
                                   [0, -1, -1, -1, -1, 0],
                                   [0, 0, -1, 0, 0, 0],
                                   [-1, 0, 0, 0, 0, 0],
                                   [0, -10, -1, -1, -1, 0],
                                   [0, 0, -1, -10, 0, 0]])

        self.rewardsDown = np.array([[0, -1, -1, -1, -1, 0],
                                     [0, 0, -1, 0, 0, 0],
                                     [-1, 0, 0, 0, 0, 0],
                                     [0, -10, -1, -1, -1, 0],
                                     [0, 0, -1, -10, 0, 0],
                                     [-1, -1, -1, 0, -1, -1]])

        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.lockPickProb = lockPickProb
        self.terminalReward = terminalReward

    def isStateTerminal(self, state):
        if state == (3, 0):
            return True
        elif state == (5, 3):
            return True
        return False

    def takeAction(self, state, action):
        retVal = []
        if (self.isStateTerminal(state)):
            return [[state, 1, self.terminalReward]]

        if action == 'left':
            reward = self.rewardsLeft[state]
            if (reward == -1):
                retVal.append([state, 1, -1])
            elif (reward == -10):
                retVal.append([(state[0], state[1] - 1), self.lockPickProb, -1])
                retVal.append([state, 1 - self.lockPickProb, -1])
            else:
                retVal.append([(state[0], state[1] - 1), 1, -1])

        if action == 'right':
            reward = self.rewardsRight[state]
            if (reward == -1):
                retVal.append([state, 1, -1])
            elif (reward == -10):
                retVal.append([(state[0], state[1] + 1), self.lockPickProb, -1])
                retVal.append([state, 1 - self.lockPickProb, -1])
            else:
                retVal.append([(state[0], state[1] + 1), 1, -1])

        if action == 'up':
            reward = self.rewardsUp[state]
            if (reward == -1):
                retVal.append([state, 1, -1])
            elif (reward == -10):
                retVal.append([(state[0] - 1, state[1]), self.lockPickProb, -1])
                retVal.append([state, 1 - self.lockPickProb, -1])
            else:
                retVal.append([(state[0] - 1, state[1]), 1, -1])

        if action == 'down':
            reward = self.rewardsDown[state]
            if (reward == -1):
                retVal.append([state, 1, -1])
            elif (reward == -10):
                retVal.append([(state[0] + 1, state[1]), self.lockPickProb, -1])
                retVal.append([state, 1 - self.lockPickProb, -1])
            else:
                retVal.append([(state[0] + 1, state[1]), 1, -1])
        for i, [nextState, prob, reward] in enumerate(retVal):
            if (self.isStateTerminal(nextState)):
                retVal[i][2] = self.terminalReward

        return retVal


class GridworldSolution:
    def __init__(self, maze, horizonLength):
        self.env = maze
        self.actionSpace = ['left', 'right', 'up', 'down']
        self.horizonLength = horizonLength
        self.DP = np.ones((self.env.gridHeight, self.env.gridWidth, self.horizonLength), dtype=float) * -np.inf

    def optimalReward(self, state, k):
        optReward = -np.inf

        time_step = self.horizonLength
        while time_step > k:
            for i in range(self.env.gridHeight):
                for j in range(self.env.gridWidth):

                    # Initializing a list to take the expected rewards for every action
                    rewards = []

                    # Looping through each action to calculate its expected reward
                    for action in self.actionSpace:
                        next_state = self.env.takeAction((i, j), action)

                        # p, q store the indices of the next state which lock picking is successful
                        # or probability to reach that stage is 1
                        p = next_state[0][0][0]
                        q = next_state[0][0][1]

                        # r, s store the indices of the next stage when lock picking fails
                        # i.e, same state as the current state
                        if len(next_state) == 2:
                            r = next_state[1][0][0]
                            s = next_state[1][0][1]

                        # This if conditional takes care of the corner case when time stamp is horizon length
                        # It makes previous values zero. We are just taking the expected value of the reward you get
                        # to transition from one state to another
                        if time_step == self.horizonLength:

                            if next_state[0][1] == 1:
                                rewards.append(next_state[0][2])
                            else:
                                rewards.append(next_state[0][1] * next_state[0][2]
                                               + next_state[1][1] * next_state[1][2])

                        # DP Algorithm. Adding the previous values to current rewards obtaining while changing states
                        else:

                            if next_state[0][1] == 1:

                                rewards.append(next_state[0][2] + self.DP[p][q][time_step])
                            else:

                                rewards.append(next_state[0][1] * (next_state[0][2] + self.DP[p][q][time_step])
                                               + next_state[1][1] * (next_state[1][2] + self.DP[r][s][time_step]))

                    # Updating the values for the next time step with max. of rewards for every action per state
                    self.DP[i][j][time_step - 1] = max(rewards)

            # Updating the time step
            time_step -= 1
            optReward = self.DP[state[0]][state[1]][k]
        return optReward


if __name__ == "__main__":
    maze = Maze()
    solution = GridworldSolution(maze, horizonLength=5)
    print(" Horizon ", solution.horizonLength)
    optReward = solution.optimalReward((2, 0), 0)
    assert optReward == 28.0, 'wrong answer'
