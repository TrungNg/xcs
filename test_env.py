from environment_controller import *
from Data_Generator import *


woods = Woods('1', True)
woods.assignPosition(3,3)
woods._display()
print(woods.detector())
print(woods.affector(-1, -1))
woods._display()
print(woods.detector())