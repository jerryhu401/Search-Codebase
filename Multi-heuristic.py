import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
import math

class Node:
    def __init__(self, state, parent, g_score):
        self.state = state
        self.parent = parent
        self.g_score = g_score

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def expand_node(self, grid):
        x, y = self.state
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if grid.connectivity == 8:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)]) 

        result = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if grid.is_traversable(nx, ny):
                result.append(Node((nx, ny), self, self.g_score + grid.grid[x][y]))

        return result
    

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
    
    def peek(self):
        return self.heap[0]

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

def heuristic_chebyshev(node, goal):
    return max(abs(node[0] - goal[0]), abs(node[1] - goal[1]))

def heuristic_octile(node, goal):
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

def imha_star(grid, start, goal, heuristics, w1, w2):
    """
    Independent Multi-Heuristic A*
    Each heuristic maintains its own independent search with its own g-values.
    """
    open = [PQ() for _ in range(len(heuristics))]  # n+1 priority queues
    open_sets = [set([start.state]) for _ in range(len(heuristics))]
    closed_set = [set() for _ in range(len(heuristics))]
    
    for i in range(len(heuristics)):
        open[i].push(start, heuristics[i](start.state, goal) * w1)
    
    while not open[0].isEmpty():
        for i in range(1, len(heuristics)):
            if not open[i].isEmpty() and open[i].peek()[0] <= w2 * open[0].peek()[0]:
                current = open[i].pop()
                open_sets[i].remove(current.state)
                closed_set[i].add(current.state)
                if current.state == goal:
                    print(i)
                    return reconstruct_path(current, start, goal), open_sets[i], closed_set[i]
                for neighbor in current.expand_node(grid):
                    g = neighbor.g_score
                    if neighbor.state not in closed_set[i]:
                        h_score = heuristics[i](neighbor.state, goal)
                        f = h_score + g
                        open[i].update((neighbor), f)
                        open_sets[i].add(neighbor.state)
            else:
                current = open[0].pop()
                open_sets[0].remove(current.state)
                closed_set[0].add(current.state)
                if current.state == goal:
                    print(i)
                    return reconstruct_path(current, start.state, goal), open_sets[0], closed_set[0]
                for neighbor in current.expand_node(grid):
                    g = neighbor.g_score
                    if neighbor.state not in closed_set[0]:
                        h_score = heuristics[0](neighbor.state, goal)
                        f = h_score + g
                        open[0].update((neighbor), f)
                        open_sets[0].add(neighbor.state)
    
    return None, open_sets[0], closed_set[0]

def smha_star(grid, start, goal, heuristics, w1, w2):
    """
    Shared Multi-Heuristic A*
    All searches share the same g-values, improving efficiency in heuristic depression regions.
    """
    open_lists = [[] for _ in range(len(heuristics) + 1)]
    came_from = {}
    g_score = {start: 0}
    open_set = set([start])
    closed_set = set()
    
    for i in range(len(heuristics)):
        heapq.heappush(open_lists[i], (heuristics[i](start, goal) * w1, start))
    
    while open_lists[0]:
        for i in range(1, len(heuristics)):
            if open_lists[i] and open_lists[i][0][0] <= w2 * open_lists[0][0][0]:
                _, current = heapq.heappop(open_lists[i])
            else:
                _, current = heapq.heappop(open_lists[0])
                closed_set.add(current)
            
            if current == goal:
                return reconstruct_path(came_from, start, goal), closed_set
            
            for neighbor in grid.expand_node(current):
                g = g_score.get(current, float('inf')) + grid.grid[neighbor]
                
                if neighbor not in g_score or g < g_score[neighbor]:
                    g_score[neighbor] = g
                    f_values = [g + heuristics[i](neighbor, goal) * w1 for i in range(len(heuristics))]
                    
                    for i in range(len(heuristics)):
                        heapq.heappush(open_lists[i], (f_values[i], neighbor))
                    open_set.add(neighbor)
                    came_from[neighbor] = current
    
    return None, open_set[0], closed_set[0]

def reconstruct_path(current, start, goal):
    """Reconstructs the path from start to goal."""
    path = []
    while current != start:
        path.append(current.state)
        current = current.parent
    path.append(start.state)
    path.reverse()
    return path

def run_search():
    width, height = 20, 20
    grid = Gridworld(width, height, 0.3, 10, connectivity=8)
    start = Node((0, 0), None, 0)
    goal = (height - 1, width - 1)
    heuristics = [heuristic_manhattan, heuristic_euclidean, heuristic_chebyshev, heuristic_octile]
    
    path, open_set, closed_set = imha_star(grid, start, goal, heuristics, 3, 3)
    if path:
        print("Path found:", path)
        grid.draw_grid(path=path, start=start.state, goal=goal, open_list=open_set, closed_list=closed_set)
    else:
        print("No path found")
        grid.draw_grid(path=path, start=start.state, goal=goal, open_list=open_set, closed_list=closed_set)

if __name__ == "__main__":
    run_search()
