class MaxHeap:
    def __init__(self):
        self.heap = []
    def parent(self, index):
        return (index - 1)//2
    def left_child(self, index):
        return 2*index + 1
    def right_child(self, index):
        return 2*index + 2
    def swap_index(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
    def insert(self, value):
        self.heap.append(value)
        current = len(self.heap) -1

        while current > 0 and self.heap[current] > self.heap[self.parent(current)]:
            self.swap_index(current, self.parent(current))    
            current = self.parent(current)

    def heapify_down(self, index):
        max_index = index
        while True:
            left_idx = self.left_child(index)
            right_idx = self.right_child(index)
            if (left_idx < len(self.heap) and self.heap[left_idx] > self.heap[max_index]):
                max_index = left_idx
            if ( right_idx < len(self.heap) and self.heap[right_idx] > self.heap[max_index]):
                max_index = right_idx
            if max_index != index:
                self.swap_index(index, max_index)
                index = max_index
            else:
                return

    def remove(self):
        if len(self.heap) == 0 :
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        max_index = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        return max_index



maxHeap = MaxHeap()

maxHeap.insert(99)
maxHeap.insert(72)
maxHeap.insert(61)
maxHeap.insert(58)

print(maxHeap.heap)

maxHeap.insert(100)

print(maxHeap.heap)
maxHeap.insert(90)
print(maxHeap.heap)
print("Removed element:", maxHeap.remove())
print(maxHeap.heap)



