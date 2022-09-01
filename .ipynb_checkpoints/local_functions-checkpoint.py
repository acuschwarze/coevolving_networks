import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import odeint, solve_ivp
from scipy.stats import beta
import time, os

def computeDiscordance(A, x):
    dm = distanceMatrix(x)
    discordance = np.sum(A*dm)
    return discordance

def getCentroidNumber(arr):
    num = len(list(set(np.round(arr*len(arr)))))
    return num

def snapshot(arr, t, coevolving=True, intention=True):
    n = nFromData(len(arr[0]), coevolving=coevolving, intention=intention)
    snapshot_data = arr[t]
    A = np.reshape(-snapshot_data[:n**2],(n,n))
    np.fill_diagonal(A,0)
    x = snapshot_data[n**2:n**2+n]
    return A, x

def nodeColors(arr, min_val=0, max_val=1, cmap='copper'):
    cm = plt.get_cmap(cmap)
    rescaled_arr = (arr-min_val)/(max_val-min_val)
    colors = [cm(x) for x in rescaled_arr]
    return colors

def drawState(arr, t, coevolving=True, intention=True, cmap1='Blues', cmap2='Reds', 
    fixed_pos=None, initial_pos=None, k=None):
    
    if k is None:
        k = 2/np.sqrt(len(arr))
    
    # get scale
    A0, x0 = snapshot(arr, 0, coevolving=coevolving, intention=intention)
    scale = np.max(x0)
    
    # get final state
    Af, xf = snapshot(arr, t, coevolving=coevolving, intention=intention)
    
    # create graph
    G = nx.from_numpy_array(Af, create_using=nx.DiGraph)
    
    # get thresholded distance mtrix
    dm = distanceMatrix(xf)
    mdm = np.mean(dm)
    dm[dm>=mdm] = 1
    dm[dm<mdm] = 0
    
    # for each edge get weight and discordance
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    #discordances = [dm[e[0],e[1]] for e in edges]
    
    # sort by weight
    edges = [e for _, e in sorted(zip(weights, edges))]
    #discordances = [d for _, d in sorted(zip(weights, discordances))]
    weights  = sorted(weights)
    
    edge_cm1 = plt.get_cmap(cmap1)
    edge_cm2 = plt.get_cmap(cmap2)
    edge_colors = [(edge_cm2(weights[i]) if dm[edges[i][0],edges[i][1]] 
                    else edge_cm1(weights[i])) for i in range(len(weights))]
    
    if fixed_pos is not None:
        new_pos = fixed_pos
    else:
        new_pos = nx.spring_layout(G, k=None, pos=initial_pos, weight='weight')
        
    nx.draw(G, new_pos, node_color=nodeColors(xf, max_val=scale), 
            edgelist=edges, edge_color=edge_colors)
    
    return new_pos
    
def plotSweepTrajectories(A=None, t0=0, t1=20.001, dt=0.001, colors=None, trajectories=True, ax=None,
    gen_alpha=None, gen_scale=None, gen_mx=None, fun1=None, fun2=None, varname1=None, varname2=None, max_iter=500):
    
    if A is None:
        # construct directed adjacency matrix of random graph
        # (from previous tries, we know that setting seed=4 gives us a strongly connected graph)
        A = nx.to_numpy_array(nx.gnp_random_graph(20, 0.15, directed=True, seed=4)) 

    n = len(A)

    if colors is None:
        colors = ['red', 'blue', 'purple', 'magenta']
        
    if gen_alpha is None:
        gen_alpha = lambda : 20 #np.random.randint(1,n-1)
    if gen_scale is None:
        gen_scale = lambda : np.random.uniform(0,1)
    if gen_mx is None:
        gen_mx = lambda : n//2
    if fun1 is None:
        fun1 = lambda Af, xf, alpha, scale, mx : np.mean(xf)
    if fun2 is None:
        fun2 = lambda Af, xf, alpha, scale, mx : np.sum(np.abs(distanceMatrix(xf)*Af))
    if varname1 is None:
        varname1 = 'mean x'
    if varname2 is None:
        varname2 = 'discordance'
    
    if ax is None:
        ax = plt.subplot(111)
    
    for it in range(max_iter):
        t_intermediate = time.time()

        alpha = gen_alpha()
        scale = gen_scale()
        mx = gen_mx()

        # set bimodal initial value
        x0 = np.zeros(n)
        x0[0:mx] = 1
        np.random.shuffle(x0)
        x0 = x0*scale

        # run simulations
        dat = simulate(A=A, x0=x0, t0=t0, t1=t1, dt=dt, degree='in', 
                       intention=False, coevolving=True, fix_indegree=True, alpha=alpha)

        var1 = np.zeros(len(dat))
        var2 = np.zeros(len(dat))
        for t in range(len(dat)):
            Af, xf = snapshot(dat, t, intention=False)
            var1[t] = fun1(Af, xf, alpha, scale, mx) 
            var2[t] = fun2(Af, xf, alpha, scale, mx) 

        cn = getCentroidNumber(xf)
        
        if trajectories:
            ax.plot(var1[::100], var2[::100], '-', color=colors[cn-1])
        ax.plot([var1[0]], [var2[0]], 'o', color=colors[cn-1])
        
        if it==0:
            print('Time per iteration:', time.time()-t_intermediate)
        
    ax.set_xlabel(varname1, fontsize=16)
    ax.set_ylabel(varname2, fontsize=16)
    return dat

################################################################################
# HELPER FUNCTIONS
################################################################################
def LDF(L, x, threshold):
    '''Get a vector $Z=(z_i)_i$ such that $z_i$ is the lowest value $x_j$ such
    that $x_j$ is $x_i$ or an in-neighbor of $x_i$ in the graph described by 
    $L$.
    
    (LDF is short for 'least drinking friend').
    '''
    
    # get adjacency matrix
    A = np.abs(L)
    np.fill_diagonal(A,1)
    A[A<threshold]=0
    
    # get xmin
    xmins = [np.min(x[(A[i]).nonzero()[0]]) for i in range(len(A))]
    
    return xmins


################################################################################    
def squaredDistanceMatrix(a):
    '''Returns square of distance matrix for an array of positions on a line.'''

    # compute distances
    distance_matrix = distanceMatrix(a)
    
    # compute squared distances
    squared_distance_matrix = distance_matrix**2
    
    return squared_distance_matrix


################################################################################    
def distanceMatrix(a):
    '''Returns the distance matrix for an array of positions on a line.'''

    # turn 1d array into 2d array
    a2d = np.array([a])
    
    # compute distances
    distance_matrix = np.abs(a2d.T-a2d)
    
    return distance_matrix


################################################################################   
def nFromData(array_length, coevolving=True, intention=True):
    '''Deduce the number of nodes from the shape of a simulation data set.'''
    
    if coevolving and intention:
        n = int(np.sqrt(array_length+1)-1)
    elif coevolving:
        n = int(0.5+np.sqrt(array_length+0.25))-1
    elif intention:
        n = array_length//2
    else:
        n = array_length
        
    return n


################################################################################            
def inputFunction(shape, t0=0, t1=0.01, max_val=0.5, alpha=2, beta=2):
    '''Return a function to be used as an external input in an ODE simulation.'''
    
    if shape in ['rect', 'rectangle', 'block']:
        nzf = lambda t: max_val
    elif shape in ['tri', 'tria', 'triangle', 'linear']:
        nzf = lambda t: max_val*(t-t0)/t1
    elif shape in ['beta']:
        nzf = lambda t: beta.pdf((t-t0)/t1, alpha, beta)
    else:
        print("Error in inputSeries:",
              "Unknown argument for 'shape'.")
        return 0
    
    # define input function
    def f(t):
        if t0<=t<=t1:
            return nzf(t)
        else:
            return 0.0

    return f


################################################################################
# THE MODEL
################################################################################
def coevolvingModel(X, t, n, intention=True, degree='in', fix_indegree=True, 
    alpha=1.0, beta=1.0, friendship_threshold=0.5,
    x_input=lambda t: 0, y_input=lambda t: 0, perspective_taking=None):
    '''Function that defines the differential dX/dt for our model of 
    coevolving network dynamics.'''
    
    # reshape input data
    L = np.reshape(X[:n**2], (n,n))
    x = X[n**2:n**2+n]
    if intention:
        y = X[-n:]
        
    # compute changes
    Delta = squaredDistanceMatrix(x)
    dL = -alpha*L*Delta
    dx = np.matmul(-L, x) - x * x_input(t)
    if intention:
        dx -=  x * y
        dy = beta*np.matmul(-L, y) - y + y_input(t) 
        if perspective_taking is not None:
            dy += (perspective_taking(t)
                   *(x-LDF(L, x, friendship_threshold)))            

    # fix in-degree of nodes
    if fix_indegree:
        dL = dL - (1-np.eye(n))*np.sum(dL, axis=-1)/(n-1)
    
    # force zero row sum in Laplace matrix
    if degree == 'in':
        np.fill_diagonal(dL, -np.sum(dL, axis=-1))
    else:
        # or force zero column sum
        np.fill_diagonal(dL, -np.sum(dL, axis=0))
        
    # collect changes for all state variables
    if intention:
        dX = np.concatenate([np.ravel(dL), dx, dy])
    else:
        dX = np.concatenate([np.ravel(dL), dx])
        
    return dX  


################################################################################
# FUNCTIONS FOR GENERATING DATA
################################################################################
def simulate(A=None, x0=None, y0=None, t0=0.0, t1=1.0, dt=0.001, 
    degree='in', fix_indegree=True, intention=False, coevolving=False, 
    to_convergence=False, max_iter=int(1E6), d_iter=int(1E3), tol=1E-3,
    alpha=0.01, beta=0.01, friendship_threshold=0.5,
    x_input=lambda t: 0, y_input=lambda t: 0, perspective_taking=lambda t: 0):
    '''Simulate `coevolvingModel`. More keyword arguments to be added.'''
    
    # set time steps
    t_array = np.arange(t0, t1, dt)
    timesteps = len(t_array)
    if to_convergence and timesteps < d_iter:
        print('Error in simulate:',
              'd_iter must be less or equal the number of simulations steps.')
        
    has_converged=False
        
    # set initial values
    if A is None:
        # get adjacency matrix of a strongly connected directed graph
        A = nx.to_numpy_array(nx.gnp_random_graph(20, 0.15, 
                                                  directed=True, seed=4))
    n = len(A)
    
    if x0 is None:
        x0 = 1 + np.sin(np.linspace(0,2*np.pi, len(A)))

    if y0 is None:
        y0 = np.zeros(len(A))

    if degree=='in':
        L0 = np.diag(np.sum(A, axis=-1)) - A
    else:
        L0 = np.diag(np.sum(A, axis=0)) - A
    
    # define initial value X0 and model dX
    if coevolving:
        if intention:
            X0 = np.concatenate([np.ravel(L0), x0, y0])
        else:
            X0 = np.concatenate([np.ravel(L0), x0])
        parameters = {'intention': intention,
                      'degree': degree,
                      'fix_indegree': fix_indegree,
                      'alpha': alpha,
                      'beta': beta,
                      'x_input': x_input,
                      'y_input': y_input,
                      'perspective_taking': perspective_taking,
                      'friendship_threshold': friendship_threshold}
        dX = lambda X, t: coevolvingModel(X, t, n, **parameters)

    else:  
        if intention:
            X0 = np.concatenate([x0, y0])
            dX = lambda X, t : np.concatenate(
                [np.matmul(-L0, X[:n]), np.matmul(-L0, X[n:])-X[n:]])
        else:
            X0 = np.array(x0)
            dX = lambda X, t: np.matmul(-L0, X)
            
    # simulate model
    for i in range(max_iter):

        data = odeint(dX, X0, t_array)
            
        if to_convergence:
            # check change over last d_iter timesteps
            change = np.sum(np.abs(data[-d_iter]-data[-1]))
            if change < tol:
                has_converged = True
                print('Simulation converged after', len(t_array))
                break
            else:
                tf = t1 + d_iter*dt
                t_array = np.arange(t0, tf, dt)
                print('Increase final time to',tf)
        else:
            break
            
    return data


################################################################################  
def simulateRectangular(x0_length=0.5, x0_value=1, x0_shuffle=False,
    y0_length=1, y0_value=0, y0_shuffle=False,
    x_input_length=1, x_input_value=0, y_input_length=1, y_input_value=0, 
    perspective_taking_length=1, perspective_taking_value=0, **kwargs):
    '''Wrapper for `simulate` that assumes one can describe all input functions 
    and initial conditions by rectangular functions.'''
    
    key_list = list(kwargs.keys())
    
    # add default value for A is none given
    if 'A' not in key_list:
        # get adjacency matrix of a strongly connected directed graph
        A = nx.to_numpy_array(nx.gnp_random_graph(20, 0.15, 
                                                  directed=True, seed=4))
        kwargs['A'] = A
        
    # get number of nodes
    n = len(kwargs['A'])

    if 'x0' not in key_list:
        # set initial condition x0
        x0 = np.zeros(n)
        x0[:int(x0_length*n)] = x0_value
        if x0_shuffle:
            np.random.shuffle(x0)
        kwargs['x0'] = x0
    
    if 'y0' not in key_list:
        # set initial condition y0    
        y0 = np.zeros(n)
        y0[:int(y0_length*n)] = y0_value
        if y0_shuffle:
            np.random.shuffle(y0)
        kwargs['y0'] = y0    
        
    if 'x_input' not in key_list:
        # set input function for x
        x_input = inputFunction('rect', t0=0, t1=x_input_length, 
            max_val=x_input_value)
        kwargs['x_input'] = x_input
        
    if 'y_input' not in key_list:
        # set input function for y
        y_input = inputFunction('rect', t0=0, t1=y_input_length, 
            max_val=y_input_value)
        kwargs['y_input'] = y_input
        
    if 'perspective_taking' not in key_list:
        # set input function for perspective-taking intervention
        perspective_taking = inputFunction('rect', t0=0, 
            t1=perspective_taking_length, 
            max_val=perspective_taking_value)
        kwargs['perspective_taking'] = perspective_taking
        
    # call 'simulate'
    data = simulate(**kwargs)
    
    return data


################################################################################
def sweep(var1, var2, fixed_parameters): # NEED TO TEST
    '''Wrapper for varying 2 simulation parameters.
    
    Parameters
    ----------
    var1 : tuple or list
       A list with 2 elements. The first element is the name of a keyword 
       argument for the function `simulate`. The second element is a list
       or array of variable values that should be considered in the 
       parameter sweep.
       
    var2 : tuple or list
       A list with 2 elements. The first element is the name of a keyword 
       argument for the function `simulate`. The second element is a list
       or array of variable values that should be considered in the 
       parameter sweep.
       
    fixed_parameters : dict
       Dictionary of keyword arguments for `simulate` or `simulateRectangular`
       
    Returns
    -------
       Array of the final states of each simulation.
    '''
    
    for i1, value1 in enumerate(var1[1]):
        fixed_parameters[var1[0]] = value1
        
        for i2, value2 in enumerate(var2[1]):
            fixed_parameters[var2[0]] = value2
            
            data = simulateRectangular(**fixed_parameters)
            if i1==0 and i2==0:
                out = np.zeros((len(var1[1]),len(var2[1]),len(data[0])))
            out[i1,i2,:] = data[-1,:]
    
    return out


################################################################################
# FUNCTIONS FOR PLOTTING DATA
################################################################################
def plotSim(data, lines=5, intention=False, coevolving=False,
    fig=None, axes=None, figtitle='', ylabel='x', legend_label='node', 
    find_nonzero=True, find_max=True):
    '''Show results of simulation in 2 plots. More keywords to be added.'''
    
    # get number of nodes from data dimension           
    n = nFromData(len(data[0]), intention=intention, coevolving=coevolving)
    
    if coevolving and intention:    
        
        # make a figure
        fig, axs = plt.subplots(3, 2, figsize=(6, 9))
        
        # plot x values
        plotSim(data[:,-2*n:-n], lines=lines, find_nonzero=find_nonzero,  
            intention=False, coevolving=False,
            fig=fig, axes=[axs[0][0], axs[0][1]], figtitle='',
            ylabel=r'$x_i$', legend_label='node')
        
        # plot y values
        plotSim(data[:,-n:], lines=lines, find_nonzero=find_nonzero, 
            intention=False, coevolving=False,
            fig=fig, axes=[axs[1][0], axs[1][1]], figtitle='',
            ylabel=r'$y_i$', legend_label='node')  
        
        # plot L values
        nd = np.where(np.ravel(np.eye(n))==0)
        plotSim(np.squeeze(data[:,nd], axis=1), lines=lines, 
            find_nonzero=find_nonzero, 
            intention=False, coevolving=False,
            fig=fig, axes=[axs[2][0], axs[2][1]], figtitle='', 
            ylabel=r'$L_{ij}$', legend_label='tie') 
        plt.subplots_adjust(top=0.77)
        
    elif coevolving:
        
        # make a figure
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        
        # plot x values
        plotSim(data[:,n**2:], lines=lines, find_nonzero=find_nonzero, 
            intention=False, coevolving=False,
            fig=fig, axes=[axs[0][0],axs[0][1]], figtitle='', 
            ylabel=r'$x_i$', legend_label='node')
        
        # plot L values
        nd = np.where(np.ravel(np.eye(n))==0)
        plotSim(np.squeeze(data[:,nd], axis=1), 
            lines=lines, find_nonzero=find_nonzero, 
            intention=False, coevolving=False,
            fig=fig, axes=[axs[1][0],axs[1][1]], figtitle='', 
            ylabel=r'$L_{ij}$', legend_label='tie')
        
    elif intention:
        
        # make a figure
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        
        # plot x values
        plotSim(data[:,-2*n:-n], lines=lines, find_nonzero=find_nonzero, 
            intention=False, coevolving=False,
            fig=fig, axes=[axs[0][0],axs[0][1]], figtitle='', 
            ylabel=r'$x_i$', legend_label='node')
        
        # plot y values
        plotSim(data[:,-n:], lines=lines, find_nonzero=find_nonzero, 
            intention=False, coevolving=False,
            fig=fig, axes=[axs[1][0],axs[1][1]], figtitle='', 
            ylabel=r'$y_i$', legend_label='node')       
    else:        
        # set indices of time series to be plotted
        indices = [[],[]]
        
        # keyword argument should be a list with 2 elements
        if np.isscalar(lines):
            lines = [int(lines), int(lines)]
            
        for i in range(2):
            
            # if an element of 'lines' is scalar it is interpreted as the 
            # number of lines to be plotted
            if np.isscalar(lines[i]):
                lines[i] = min([data.shape[i], lines[i]])
                
                # select indices of time series to be plotted
                if find_max:
                    
                    # find time series for which the system state undergoes
                    # the largest change; use those as indices for plotting
                    indices[i] = np.sort(
                        np.argsort(np.abs(data[0]-data[-1]))[-lines[i]:])
                    
                elif find_nonzero:
                    
                    # find indices corresponding to non-zero time series
                    nzs = np.nonzero(np.sum(np.abs(data), axis=0))[0]
                    
                    # select a number of those indices (that number is 
                    # specified bt the 'lines' keyword argument)
                    selection = np.array(
                        np.linspace(0, len(nzs)-1, lines[i]), dtype=int)
                    
                    # save as indices to be used for plotting
                    indices[i] = [nzs[s] for s in selection]
                    
                else:
                    # select the specified number of indices for plotting
                    # such that they are equidistantly spaced over the list
                    # of state variables
                    indices[i] = np.array(
                        np.linspace(0, data.shape[i]-1, lines[i]), dtype=int)
            else:
                # if an element of 'lines' is a list it is interpreted as the 
                # as the indices of the lines to be plotted                   
                indices[i] = lines[i]
            
        # plot
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig)
            
        if axes is None: 
            axes = (plt.subplot(121), plt.subplot(122))
        else:
            ax1 = axes[0]
            ax2 = axes[1]
        
        # set colors for snap shot plots
        colors = [ [1-v,v/2,v] for v in 
                  np.linspace(0, 0.8, len(indices[0]))][::-1]
        
        # plot snapshot plots
        for ci, i in enumerate(indices[0]):
            axes[0].plot(data[i,:], marker='d', label='t='+str(i), 
                color=colors[ci], linestyle=':')
            
        axes[0].set_xlabel(legend_label+' index')
        axes[0].set_ylabel(ylabel)    
        axes[0].legend(bbox_to_anchor=(-0.05,1), loc=3, ncol=2)

        # plot time series
        for i in indices[1]:
            axes[1].plot(data[:,i], '-', label=legend_label+' '+str(i))
        axes[1].set_xlabel('time')
        axes[1].set_ylabel(ylabel)
        axes[1].legend(bbox_to_anchor=(-0.05,1), loc=3, ncol=2)

        plt.subplots_adjust(wspace=0.4, top=0.7, hspace=1.1)
        plt.suptitle(figtitle, fontsize=16)