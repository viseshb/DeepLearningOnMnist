class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self. right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        new_node = Node(value)
        temp = self.root
        if self.root is None:
            self.root = new_node
            return True
        while True:
            if value == temp.value:
                return False
            if value < temp.value:
                if temp.left is None:
                    temp.left = new_node    
                    return True
                temp = temp.left
            else:
                if temp.right is None:
                    temp.right = new_node
                    return True
                temp = temp.right
    # def __print_in_order(self, node):
    #     if node:
    #        self.__print_in_order(node.left)
    #        print(node.value, end=" ")
    #        self.__print_in_order(node.right)

    # def print_tree(self):
    #     if self.root is None:
    #        print("Tree is empty.")
    #     else:
    #        self.__print_in_order(self.root)
    #        print()

    def __r_Contains(self, node, value):
        if node == None:
            return False
        if value == node.value:
            return True
        if value < node.value:
            return self.__r_Contains(node.left, value)
        if value > node.value:               
            return self.__r_Contains(node.right, value)

    def r_contains(self, value):
        return self.__r_Contains(self.root, value)     



myTree = BST()
myTree.insert(47)
myTree.insert(21)
myTree.insert(76)
myTree.insert(18)
myTree.insert(27)
myTree.insert(52)
myTree.insert(82)

print('BST Contains 52:', myTree.r_contains(52))
print('BST Contains: 40:', myTree.r_contains(40))
myTree.print_tree()