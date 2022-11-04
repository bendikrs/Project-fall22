import numpy as np

old_map    = np.array([0, 1,0.5, 1, 0.5,  0,  1])
map        = np.array([0, 1,  1, 0,   1,0.5,0.5])
#fasit_map = np.array([0, 1,  1, 1,   1])

# Where the map is 0.5, the new map should be same value as the old_map
new_map = np.where(map == 0.5, old_map, map)

print(new_map)