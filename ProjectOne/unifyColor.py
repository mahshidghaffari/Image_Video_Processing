from random import random

import numpy as np


def flood_recursive(matrix):
    width = len(matrix)
    height = len(matrix[0])
    #pick a random starting point
    start_x = np.random.randint(0,width-1)
    start_y = np.random.randint(0,height-1)
    start_color = matrix[start_x][start_y]

    def fill(x,y,start_color,color_to_update):
        #if the square is not the same color as the starting point
        if matrix[x][y] != start_color:
            print(1)
            return
        #if the square is not the new color
        elif matrix[x][y] == color_to_update:
            print(2)
            return
        else:
            print("hi")
            #update the color of the current square to the replacement color
            matrix[x][y] = color_to_update
            neighbors = [(x-1,y),(x+1,y),(x-1,y-1),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x,y-1),(x,y+1)]
            for n in neighbors:
                if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1:
                    fill(n[0],n[1],start_color,color_to_update)






    fill(start_x, start_y, start_color, 9)

    return matrix

