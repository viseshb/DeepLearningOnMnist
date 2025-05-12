from collections import deque

def bfs_directed(graph, start):
    visited = set()
    queue = deque([start])
    bfs_order = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            bfs_order.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return bfs_order

graph1 = {
    'A': ['B', 'D'],
    'B': ['A', 'C', 'D', 'E'],
    'C': ['B', 'E'],
    'D': ['A', 'B', 'E', 'F'],
    'E': ['B', 'C', 'D', 'F', 'G'],
    'F': ['D', 'E', 'G'],
    'G': ['E', 'F']
}

graph2 = {
    1: [2, 3],
    2: [3, 4, 5],
    3: [5],
    4: [6],
    5: [4,6],
    6: []
}

graph3 = {
    'A': ['B', 'F', 'I'],
    'B': ['A','E', 'C'],
    'C': ['B','E', 'D'],
    'D': ['C', 'G', 'H'],
    'E': ['B', 'C', 'G'],
    'F': ['A','G'],
    'G': ['D','E', 'F', 'H'],
    'H': ['D'],
    'I': ['A']
}

graph4 = {
    'A': ['C', 'D'],
    'B': ['C'],
    'C': ['F','G'],
    'D': ['E','G', 'H'],
    'E': [],
    'F': ['B'],
    'G': [],
    'H': ['E']
    
}
# Run DFS
bfs_order = bfs_directed(graph1, 'A')
print("BFS Order Q1:", bfs_order)
print("-----------------------")
bfs_order = bfs_directed(graph2, 1)
print("BFS Order Q2:", bfs_order)
print("-----------------------")
bfs_order = bfs_directed(graph3, 'A')
print("BFS Order Q3:", bfs_order)
print("-----------------------")
bfs_order = bfs_directed(graph4, 'A')
print("BFS Order Q4:", bfs_order)

