import matplotlib.pyplot as plt
from matplotlib import cm

def plot_height(height,L=None,color_map='terrain'):
    # note that this function produces an image, but does not return anything

    # if no L is supplied, set it to be large enough to fit all coordinates
    if not L:
        Lmax = max(max(height.keys()))+1
        Lmin = min(min(height.keys()))
    else:
        Lmax = L
        Lmin = 0
    
    # loop over all coordinates, and set any that are not present to be 0
    for x in range(Lmin,Lmax):
        for y in range(Lmin,Lmax):
            if (x,y) not in height:
                height[x,y] = 0
    
    # put the heights in a matplotlib-friendly form
    z = [ [ height[x,y] for x in range(Lmin,Lmax)] for y in range(Lmin,Lmax) ]
 
    # plot it as a contour plot, using the supplied colour map
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    cs = ax.contourf(z,25,vmin=0,vmax=1,cmap=cm.get_cmap(color_map))
    plt.axis('off')
    plt.show()