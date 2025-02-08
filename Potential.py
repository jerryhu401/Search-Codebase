import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
import math

#type annotate
#MAKE node class to replace cost so far and came from.
#Should have: state = (x, y), parent_node, cost_so_far
#hash, str, comparison

class Node:
    def __init__(self, state, parent, g_score):
        self.state = state
        self.parent = parent
        self.g_score = g_score

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state
    

class PQ:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            PQ.push(self, item, priority)


class Gridworld:
    def __init__(self, width, height, obstacle_prob, max_cost, connectivity):
        self.width = width
        self.height = height
        self.grid = np.random.randint(1, max_cost + 1, (height, width))
        self.connectivity = connectivity
        self.grid[np.random.rand(height, width) < obstacle_prob] = -1  

    def is_within_bounds(self, x, y):
        return 0 <= x < self.height and 0 <= y < self.width

    def is_traversable(self, x, y):
        return self.is_within_bounds(x, y) and self.grid[x, y] != -1

    def expand_node(self, node):
        x, y = node.state
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if self.connectivity == 8:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)]) 

        result = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_traversable(nx, ny):
                result.append(Node((nx, ny), node, node.g_score + grid.grid[x][y]))

        return result

    def draw_grid(self, path=None, start=None, goal=None, open_list=None, closed_list=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        grid_display = np.full_like(self.grid, fill_value=0.0, dtype=float)  # All cells white

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == -1:
                    grid_display[i, j] = 1.0  # Obstacles (black)

        ax.imshow(grid_display, cmap=plt.cm.binary, vmin=0, vmax=1)  # Binary colormap (0 -> white, 1 -> black)

        # Overlay path cells with green
        if path:
            for (x, y) in path:
                if (x, y) != start and (x, y) != goal:
                    ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='green', alpha=0.5))  # Path in green
                elif (x, y) == start:
                    ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='blue', alpha=0.5))  # start in blue
                else:
                    ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='red', alpha=0.5))  # goal in red

        # Overlay open list cells with yellow
        if open_list:
            for (x, y) in open_list:
                ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='yellow', alpha=0.5))

        # Overlay closed list cells with gray
        if closed_list:
            for (x, y) in closed_list:
                ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='gray', alpha=0.5))

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == -1:
                    ax.text(j, i, 'o', ha='center', va='center', color='red', fontsize=8)  # Obstacles
                else:
                    ax.text(j, i, int(self.grid[i, j]), ha='center', va='center', color='black', fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.show()

def heuristic_manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def heuristic_euclidean(node, goal):
    return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5

def linear(C, h_n, g_n):
    if h_n == 0:
        return 0
    if ((C - g_n) <= 0):
        return math.inf
    return h_n/(C - g_n)

def additive(C, h_n, g_n):
    return h_n + g_n

def potential_search(grid, start, goal, budget, cost_model, h):
    """
    Potential Search with support for different cost models: additive, linear relative, and general invertible.
    """
    open_list = PQ()
    open_list.push((start), 0)
    open_set = {start.state}
    closed_set = set()

    while open_list:
        current_node = open_list.pop()
        open_set.remove(current_node.state)
        closed_set.add(current_node.state)

        if current_node.state == goal:
            path = []
            while current_node != start:
                path.append(current_node.state)
                current_node = current_node.parent
            path.append(start.state)
            path.reverse()
            return path, open_set, closed_set

        for neighbor in grid.expand_node(current_node):
            #compute g_score in expand_node
            g_score = neighbor.g_score

            if neighbor.state not in closed_set:
                h_score = h(neighbor.state, goal)
                potential = cost_model(budget, h_score, g_score)
                open_list.update((neighbor), potential)
                open_set.add(neighbor.state)

    return None, open_set, closed_set  # No path found


if __name__ == "__main__":
    width, height = 20, 20
    grid = Gridworld(width, height, 0.3, 10, connectivity=8)

    start = Node((0, 0),None,0)
    goal = (height - 1, width - 1)
    budget = 80

    while not grid.is_traversable(*(start.state)) or not grid.is_traversable(*goal):
        grid = Gridworld(width, height, 0.3, 10, connectivity=8)

    cost_model = "linear_relative" 
    path, open_set, closed_set = potential_search(grid, start, goal, budget, linear, heuristic_manhattan)

    if path:
        print("Path found:", path)
        grid.draw_grid(path=path, start=start.state, goal=goal, open_list=open_set, closed_list=closed_set)
    else:
        print("No path found")
        grid.draw_grid(path=path, start=start.state, goal=goal, open_list=open_set, closed_list=closed_set)
