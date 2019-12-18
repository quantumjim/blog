from qiskit import QuantumCircuit, execute, Aer
from math import pi
import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib import cm

def plot_height(height,color_map='terrain'):
    # note that this function produces an image, but does not return anything

    Lmax = max(max(height.keys()))+1
    Lmin = min(0,min(min(height.keys())))
    
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
    
def make_line ( length ):
    # determine the number of bits required for at least `length` bit strings
    n = int(np.ceil(np.log(length)/np.log(2)))
    # start with the basic list of bit values
    line = ['0','1']
    # each application of the following process double the length of the list,
    # and of the bit strings it contains
    for j in range(n-1):
        # this is done in each step by first appending a reverse-ordered version of the current list
        line = line + line[::-1]
        # then adding a '0' onto the end of all bit strings in the first half
        for j in range(int(len(line)/2)):
            line[j] += '0'
        # and a '1' onto the end of all bit strings in the second half
        for j in range(int(len(line)/2),int(len(line))):
            line[j] += '1'
    return line

def make_grid(L):
    
    line = make_line( L )
    
    grid = {}
    for x in range(L):
        for y in range(L):
            grid[ line[x]+line[y] ] = (x,y)
    
    return grid

def height2circuit(height,grid):
    
    n = len( list(grid.keys())[0] )
    
    Lmax = max(max(height.keys()))+1
    Lmin = min(0,min(min(height.keys())))
        
    state = [0]*(2**n)
    
    H = 0
    for bitstring in grid:
        (x,y) = grid[bitstring]
        if (x,y) in height:
            h = height[x,y]
            state[ int(bitstring,2) ] = np.sqrt( h )
            H += h
        
    for j,amp in enumerate(state):
        state[ j ] = amp/np.sqrt(H)
                
    qc = QuantumCircuit(n,n,name=str((Lmax,Lmin)))
    qc.initialize( state, qc.qregs )
        
    return qc

def circuit2height(qc,grid,backend,shots=None,log=False):
    
    # get the number of qubits from the circuit
    n = qc.n_qubits
    
    (Lmax,Lmin) = eval(qc.name)
    
    # construct a circuit to perform z measurements
    meas = QuantumCircuit(n,n)
    for j in range(n):
        meas.measure(j,j)
        
    # if no shots value is supplied use 4**n by default (unless that is too small)
    if not shots:
        shots = max(4**n,8192)

    #run the circuit on the supplied backend
    counts = execute(qc+meas,backend,shots=shots).result().get_counts()
    
    # determine max and min counts values, to use in rescaling
    if log: # log=True uses the log of counts values, instead of the values themselves
        min_h = np.log( 1/10 ) # fake small counts value for results that didn't appear
        max_h = np.log( max( counts.values() ) )
    else:
        min_h = 0
        max_h = max( counts.values() )   
    
    # loop over all bit strings in `counts`, and set the corresponding value to be
    # the height for the corresponding coordinate. Values are rescaled to ensure
    # that the biggest height is 1, and that no height is less than zero.
    height = {(x,y):0 for x in range(Lmin,Lmax) for y in range(Lmin,Lmax)}
    for bitstring in counts:
        if bitstring in grid:
            if log: # log=True uses the log of counts values, instead of the values themselves
                height[ grid[bitstring] ] = ( np.log(counts[bitstring]) - min_h ) / (max_h-min_h)
            else:
                height[ grid[bitstring] ] = ( counts[bitstring] - min_h ) / (max_h-min_h)
    
    return height

def shuffle_height (height,grid):
    
    # determine the number of qubits
    n = int( np.log(len(grid))/np.log(2) )
    
    # randomly choose a way to shuffle the bit values in the string
    shuffle = [j for j in range(n)]
    random.shuffle(shuffle)
    
    # for each bit string, determine and record the pair of positions
    # * `pos`: the position correspoding to the bit string in the given `grid`
    # * `new_pos`: the position corresponding to the shuffled version of the bit string
    remap = {}
    for bitstring in grid:
        
        shuffledstring = ''.join([bitstring[j] for j in shuffle])

        pos = grid[bitstring]
        new_pos = grid[shuffledstring]
        
        remap[pos] = new_pos
        
    # create and return `new_height`, in which each point is moved from `pos` to `new_pos`
    new_height = {}
    for pos in height:
        new_height[remap[pos]] = height[pos]
        
    return new_height