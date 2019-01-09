""""" V will be the number of vertices. inf will be a variable for infinity. 
This will be added to a matrix where no edge exists between vertices """""

V = 7
inf = 1000000000

def floydAlgorithim(graph):
    """" dist[][] will be the output matrix that will display
        the shortest distances between every pair of vertices """""
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

    """ This loop adds vertices one by one to the set of intermediate 
    vertices. 
    
    Before start of an iteration, we have shortest distances of paths
    between all pairs of vertices checked so far in the intermediate set
    {0, 1, 2, .. k-1} as intermediate vertices.
     
    At the end of each iteration, vertex k is 
    added to the set of intermediate vertices and the 
    set becomes {0, 1, 2, .. k} 
    """
    print("\nThe following matrices show the progression of finding the shortest distance matrix\n")

    for k in range(V):

        # pick all vertices as source one by one
        for i in range(V):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(V):
                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                dist[i][j] = min(dist[i][j],
                                 dist[i][k] + dist[k][j]
                                 )

        # in python 3 and higher, * flattens multidemensional objects
        # sep determines how to break after each segment of the object

        print(*dist, sep='\n')
        print()

# row i represents the "from" vertex
# column j represents the "to" vertex
# G(i,j) is the weight from vertex i to vertex j
graph = [[0, 2, inf, 1, 8],
         [6, 0, 3, 2, inf],
         [inf, inf, 0, 4, inf],
         [inf, inf, 2, 0, 3],
         [3, inf, inf, inf, 0]]
# Print the solution
print(floydAlgorithim(graph))


""" This code is contributed by Nikhil Kumar Singh(nickzuck_007)
I pulled most of this code off of https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/ 
and mainly just rebuilt the graph to the specifications of the exam problem"""