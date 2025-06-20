class MinHeap:
    def __init__(self):
        self.heap = []
    def parent(self, index):
        return (index -1)//2 
    def left_child(self, index):
        return index*2 +1
    def right_child(self,index):
        return index*2 +2
    def swap_index( self, index1, index2):
        self.heap[index1], self.heap[index2] =self.heap[index2], self.heap[index1]
    def insert(self, value):
        self.heap.append(value)
        current = len(self.heap) - 1

        while current > 0 and self.heap[current] < self.heap[self.parent(current)]:
            self.swap_index(current, self.parent(current))
            current = self.parent(current)

    def heapify_up(self, index):
        min_index = index
        while True:
            left = self.left_child(index)
            right = self.right_child(index)
            if (left < len(self.heap) and self.heap[left] < self.heap[min_index] ):
                min_index = left
            if (right < len(self.heap) and self.heap[right] < self.heap[min_index]):
                min_index = right

            if min_index != index:
                self.swap_index(index, min_index)
                index = min_index 
            else:
                return

    def remove(self):
        if(len(self.heap) == 0):
            return None

        if (len(self.heap) == 1):
            return self.heap.pop()

        min_index = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_up(0)

        return min_index            




myHeap = MinHeap()
myHeap.insert(99)
myHeap.insert(72)
myHeap.insert(61)
myHeap.insert(58)    
print(myHeap.heap)      
myHeap.insert(1)

print(myHeap.heap)
myHeap.insert(90)
print(myHeap.heap)
print("Element removed:", myHeap.remove())
print(myHeap.heap)


