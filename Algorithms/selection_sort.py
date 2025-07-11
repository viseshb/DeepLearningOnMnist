def selection_sort(arr):
    n = len(arr)
    for i in range(n-1): #if n= 5, then i goes upto 0,1,2,3
        min_index = i
        for j in range(i+1, n): #if n = 5 then j goes upto 0,1,2,3,4
            if arr[j] < arr[min_index]:
               min_index = j
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]

arr = [5,4,3,2,1]
selection_sort(arr)
print(arr)            

