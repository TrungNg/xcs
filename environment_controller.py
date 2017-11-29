'''
Name: environment_controller.py
Author: Bao-Trung Nguyen
Created: July 14, 2017
Description:  Generate and manipulate environment for multi-step problems.
'''

#Import Required Modules---------------
import random
import math
#--------------------------------------

#Define Constants----------------------
_TREE = 0b11
_FOOD = 0b01
_PATH = 0b00
MAX_STEPS = 1
_GOAL_REWARD = 1000
_NO_REWARD = 0
_PUNISHMENT = -1000
#--------------------------------------


# Note: within the class, to access matrix self.env, y coordinate before x
# ------but wrapper interface of Environment use x before y
class Environment:
    def __init__(self, display=False):
        """ Initialize new general Environment """
        self.generateNewPosition()
        if display:
            self._display()


    def detector(self, x = None, y = None):
        """ Detect agent's observation of surrounded environment. """
        if x == None:
            x = self._x
        if y == None:
            y = self._y
        observation = []
        for i in range(-MAX_STEPS, MAX_STEPS + 1):
            for j in range(-MAX_STEPS, MAX_STEPS + 1):
                if i != 0 or j != 0:
                    neighbor_x = x + i
                    neighbor_y = y + j
                    if neighbor_x > self._w - 1:
                        neighbor_x = i - 1
                    elif neighbor_x < 0:
                        neighbor_x = self._w + i
                    if neighbor_y > self._h - 1:
                        neighbor_y = j - 1
                    elif neighbor_y < 0:
                        neighbor_y = self._h + j
                    observation.append( self.env[neighbor_y][neighbor_x] )
        return observation


    def affector(self, delta_x, delta_y):
        """ move agent to a neighborhood position """
        if math.fabs(delta_x) > MAX_STEPS or math.fabs(delta_y) > MAX_STEPS:
            raise RuntimeError("illegal step")

        next_y = self._y + delta_y
        next_x = self._x + delta_x
        if next_x > self._w - 1:
            next_x = delta_x - 1
        elif next_x < 0:
            next_x = self._w + delta_x
        if next_y > self._h - 1:
            next_y = delta_y - 1
        elif next_y < 0:
            next_y = self._h + delta_y
        if self.env[next_y][next_x] != _PATH and self.env[next_y][next_x] != _FOOD:
            return [_NO_REWARD, "CONT"]
        else:
            self._x = next_x
            self._y = next_y
            if self.env[next_y][next_x] == _FOOD:
                return [_GOAL_REWARD, "EOP"]
            return [_NO_REWARD, "CONT"]


    def generateNewPosition(self):
        """ Generate new position for agent """
        self._x = int( random.random() * self._w )
        self._y = int( random.random() * self._h )
        while self.env[self._y][self._x] != _PATH:
            self._x = int( random.random() * self._w )
            self._y = int( random.random() * self._h )
        return self._x, self._y


    def listAllObservationsAndPositions(self):
        """ return list and number of distinct positions (along with observation) agent can have """
        observation_position = []
        count = 0
        for i in range(self._h):
            for j in range(self._w):
                if self.env[i][j] == _PATH:
                    observation = self.detector(j, i)
                    observation_position.append( [ observation, [j, i] ] )
                    count += 1
        return observation_position, count


    def listOfActions(self):
        """ return a list and number of all possible actions """
        actions = []
        count = 0
        for i in range(-MAX_STEPS, MAX_STEPS + 1):
            for j in range(-MAX_STEPS, MAX_STEPS + 1):
                if i != 0 or j != 0:
                    actions.append((i,j))
                    count += 1
        return actions, count


    def getAgentPosition(self):
        """ return current position of agent in environment """
        return self._x, self._y


    def maxX(self):
        """ Return width of environment or width of the pattern in case of infinitely repeated environment """
        return self._w - 1


    def maxY(self):
        """ Return height of environment or height of the pattern in case of infinitely repeated environment """
        return self._h - 1


    def assignPosition(self, x, y, silence = True):
        """ Assign a new position for agent """
        if x not in range(self._w) or y not in range(self._h) or self.env[y][x] != _PATH:
            if not silence:
                print("Error: assignPosition inapproprate coordinates, position not assigned")
            return False
        self._x = x
        self._y = y
        return True


    def _display(self):
        for i in range(self._h):
            for j in range(self._w):
                if self._x == j and self._y == i:
                    obj = 'A'
                elif self.env[i][j] == _TREE:
                    obj = 'T'
                elif self.env[i][j] == _FOOD:
                    obj = 'F'
                else:
                    obj = '_'
                print( obj + '\t', end='' )
            print('\n')


# inherited class from Environment
class Maze (Environment):
    def __init__(self, version='5', display=False):
        """ Initialize new Maze Environment """
        options={'5':self._maze5_generate_inside_objects,
                 '6':self._maze6_generate_inside_objects}
        if version[0] not in options.keys():
            raise RuntimeError("Maze.__init__() found wrong environment_name")
            return

        self._w, self._h = 9, 9;
        self.env = [[_PATH for i in range(self._w)] for j in range(self._h)]
        for i in range(self._h):
            for j in range(self._w):
                if j==0 or j==self._w-1 or i==0 or i==self._h-1:
                    self.env[i][j] = _TREE

        options[version[0]]()
        super().__init__(display)


    def _maze5_generate_inside_objects(self):
        self.env[1][7] = _FOOD

        self.env[2][3] = _TREE
        self.env[2][5] = _TREE
        self.env[2][6] = _TREE
        self.env[3][2] = _TREE
        self.env[4][4] = _TREE
        self.env[4][5] = _TREE
        self.env[5][2] = _TREE
        self.env[5][4] = _TREE
        self.env[5][7] = _TREE
        self.env[6][2] = _TREE
        self.env[6][5] = _TREE
        self.env[7][6] = _TREE


    def _maze6_generate_inside_objects(self):
        self.env[1][7] = _FOOD

        self.env[1][6] = _TREE
        self.env[2][3] = _TREE
        self.env[2][5] = _TREE
        self.env[2][6] = _TREE
        self.env[3][2] = _TREE
        self.env[4][4] = _TREE
        self.env[4][5] = _TREE
        self.env[5][2] = _TREE
        self.env[5][4] = _TREE
        self.env[5][7] = _TREE
        self.env[6][2] = _TREE
        self.env[6][5] = _TREE
        self.env[7][6] = _TREE


# inherited class from Environment
class Woods (Environment):
    def __init__(self, version='1', display=False):
        options={'1':self._woods1_generate_inside_objects,
                 '2':self._woods2_generate_inside_objects}
        if version[0] not in options.keys():
            raise RuntimeError("Woods.__init__() found wrong environment version")
            return

        self._w, self._h = 5, 5;
        self.env = [[_PATH for i in range(self._w)] for j in range(self._h)]

        options[version[0]]()
        super().__init__(display)


    def _woods1_generate_inside_objects(self):
        for i in range(2, 5):
            for j in range(0, 3):
                self.env[i][j] = _TREE
        self.env[2][2] = _FOOD


    def _woods2_generate_inside_objects(self):
        for i in range(2, 5):
            for j in range(0, 3):
                self.env[i][j] = _TREE
        self.env[2][2] = _FOOD


