def dfs_stack(graph, start):
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            # Push neighbors in reverse alphabetical order
            for neighbor in sorted(graph[node], reverse=True):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return order

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
    'G': ['D','E', 'F'],
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
graph5 = {
    0 : [1, 3, 4],
    1 : [0, 5, 6],
    2 : [4, 5],
    3 : [0, 7],
    4 : [0, 2, 6, 7],
    5 : [1, 2, 6, 7],
    6 : [1, 4, 5, 7],
    7 : [3, 4, 5, 6]    
}

# Run DFS
dfs_order = dfs_stack(graph1, 'A')
print("DFS Order Q1:", dfs_order)
print("-----------------------")
dfs_order = dfs_stack(graph2, 1)
print("DFS Order Q2:", dfs_order)
print("-----------------------")
dfs_order = dfs_stack(graph3, 'A')
print("DFS Order Q3:", dfs_order)
print("-----------------------")
dfs_order = dfs_stack(graph4, 'A')
print("DFS Order Q4:", dfs_order)
print("-----------------------")
dfs_order = dfs_stack(graph5, 0)
print("DFS Order Q5:", dfs_order)  
