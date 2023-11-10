import numpy as np
from state import next_state, solved_state
from location import next_location, solved_location
from util import util

location_dict = {
    1: [0, 0, 0],
    2: [0, 0, 1],
    3: [0, 1, 0],
    4: [0, 1, 1],
    5: [1, 0, 0],
    6: [1, 0, 1],
    7: [1, 1, 0],
    8: [1, 1, 1],
}


class cube:
    def __init__(self, state, cost, sequence, location=None) -> None:
        self.state = state
        self.cost = cost
        self.sequence = sequence
        self.location = location

    def __iter__(self):
        return iter(self.state)


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
        return A_STAR(init_state, init_location)

    elif method == "BiBFS":
        return Bi_BFS(init_state)

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


def A_STAR(start_state, start_location):
    visited = dict()
    actions = []
    fringe = util.PriorityQueue()
    expanded_nodes = 0
    fringe.push(
        cube(
            start_state,
            heuristic(start_location) + len(actions),
            actions,
            start_location,
        ),
        heuristic(start_location) + len(actions),
    )
    while not fringe.isEmpty():
        current_node = fringe.pop()
        expanded_nodes += 1
        if np.array_equal(current_node.state, solved_state()):
            solution_info(visited, current_node.sequence, expanded_nodes)
            return current_node.sequence
        elif to_tuple(current_node.state) not in visited:
            visited[to_tuple(current_node.state)] = current_node.cost
            for i in range(12):
                nextState = next_state(current_node.state, i + 1)
                nextLocation = next_location(current_node.location, i + 1)
                nextCost = heuristic(nextLocation) + len(current_node.sequence) + 1
                if to_tuple(nextState) not in visited:
                    fringe.push(
                        cube(
                            nextState,
                            nextCost,
                            current_node.sequence + [i + 1],
                            nextLocation,
                        ),
                        nextCost,
                    )
                else:
                    base_state_to_current_node_cost = visited[to_tuple(nextState)]
                    if nextCost < base_state_to_current_node_cost:
                        fringe.update(
                            cube(
                                nextState,
                                nextCost,
                                current_node.sequence + [i + 1],
                                nextLocation,
                            ),
                            nextCost,
                        )

    return []


def heuristic(location):
    heuristic_estimation = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x, y, z = location_dict[location[i][j][k]]
                heuristic_estimation += abs(x - i) + abs(y - j) + abs(z - k)

    return heuristic_estimation / 4


def Bi_BFS(startState):
    fringe1 = util.Queue()
    fringe2 = util.Queue()
    visited1 = dict()
    visited2 = dict()
    fringe1.push(cube(startState, 0, []))
    fringe2.push(cube(solved_state(), 0, []))
    expanded_nodes = 0

    while not fringe1.isEmpty() or not fringe2.isEmpty():
        cube1: cube = fringe1.pop()
        cube2: cube = fringe2.pop()
        expanded_nodes += 2

        if np.array_equal(cube1.state, solved_state()) and np.array_equal(
            cube2.state, solved_state()
        ):
            return []
        if to_tuple(cube1.state) in visited2:
            actions = Bi_BFS_actions_appending(
                cube1.sequence, visited2[to_tuple(cube1.state)]
            )
            return actions

        if to_tuple(cube2.state) in visited1:
            actions = Bi_BFS_actions_appending(
                visited1[to_tuple(cube2.state)], cube2.sequence
            )
            return actions

        if to_tuple(cube1.state) not in visited1:
            visited1[to_tuple(cube1.state)] = cube1.sequence
            for i in range(12):
                nextState1 = next_state(cube1.state, i + 1)
                if to_tuple(nextState1) not in visited1:
                    fringe1.push(
                        cube(nextState1, cube1.cost + 1, cube1.sequence + [i + 1])
                    )

        if to_tuple(cube2.state) not in visited2:
            visited2[to_tuple(cube2.state)] = cube2.sequence
            for i in range(12):
                nextState2 = next_state(cube2.state, i + 1)
                if to_tuple(nextState2) not in visited2:
                    fringe2.push(
                        cube(nextState2, cube2.cost + 1, cube2.sequence + [i + 1])
                    )

    return []


Bi_BFS_actions_appending = lambda actions1, actions2: actions1 + [
    action + 6 if action <= 6 else action - 6 for action in actions2[::-1]
]


solution_info = lambda explored, actions, expanded_nodes: (
    print(
        "Depth of the solution path:",
        len(actions),
        "\nTotal number of nodes explored:",
        len(explored),
        "\nTotal number of expanded nodes:",
        expanded_nodes,
    )
)


# use this function to convert a numpy array to a tuple so that it can be used as a key in a dictionary or set
def to_tuple(array):
    return tuple(map(tuple, array))
