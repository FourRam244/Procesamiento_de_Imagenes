# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:00:25 2022

@author: cesar
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

im = Image.open('Lenna.png')

plt.imshow(im)

ax = plt.gca()

rect = patches.Rectangle((0,0),
                 510,
                 510,
                 linewidth=5,
                 edgecolor='cyan',
                 fill = False)
ax2 = plt.gca()

for i in range(0,300,20):
    rect2 = patches.Rectangle((0,i),
                 510,
                 10,
                 facecolor='red',
                 fill = True)
    #ax2.add_patch(rect2)
    
    ax3 = plt.gca()

for j in range(300,510,20):
    rect3 = patches.Rectangle((0,j),
                 510,
                 10,
                 facecolor='blue',
                 fill = True)
    #ax3.add_patch(rect3)

ax.add_patch(rect) #borde






    

    
plt.show()