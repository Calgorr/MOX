import numpy as np
from state import next_state, solved_state
from location import next_location
from util import util

# from cube import cube


class cube:
    def __init__(self, state, cost, sequence) -> None:
        self.state = state
        self.cost = cost
        self.sequence = sequence


def solve(init_state, init_location, method):
    """
    Solves the given Rubik's cube using the selected search algorithm.

    Args:
        init_state (numpy.array): Initial state of the Rubik's cube.
        init_location (numpy.array): Initial location of the little cubes.
        method (str): Name of the search algorithm.

    Returns:
        list: The sequence of actions needed to solve the Rubik's cube.
    """

    # instructions and hints:
    # 1. use 'solved_state()' to obtain the goal state.
    # 2. use 'next_state()' to obtain the next state when taking an action .
    # 3. use 'next_location()' to obtain the next location of the little cubes when taking an action.
    # 4. you can use 'Set', 'Dictionary', 'OrderedDict', and 'heapq' as efficient data structures.

    if method == "Random":
        return list(np.random.randint(1, 12 + 1, 10))

    elif method == "IDS-DFS":
        return IDFS(init_state)

    elif method == "A*":
        ...

    elif method == "BiBFS":
        ...

    else:
        return []


def IDFS(start_state):
    cost_limit = 1
    while True:
        fringe = util.Stack()
        visited = set()
        fringe.push(cube(start_state, 0, list()))
        while not fringe.isEmpty():
            current_node = fringe.pop()
            if np.array_equal(current_node.state, solved_state()):
                return current_node.sequence
            if (
                current_node.cost <= cost_limit
                and to_tuple(current_node.state) not in visited
            ):
                visited.add(to_tuple(current_node.state))
                for i in range(12):
                    nextState = next_state(current_node.state, i + 1)
                    if to_tuple(nextState) not in visited:
                        fringe.push(
                            cube(
                                nextState,
                                current_node.cost + 1,
                                current_node.sequence + [i + 1],
                            )
                        )
        cost_limit += 1


# use this function to convert a numpy array to a tuple so that it can be used as a key in a dictionary or set
def to_tuple(array):
    return tuple(map(tuple, array))
