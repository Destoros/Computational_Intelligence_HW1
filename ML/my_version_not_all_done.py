#Filename: HW1_skeleton.py

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats



#--------------------------------------------------------------------------------
# Assignment 1
def main():
    
    # choose the scenario
    scenario = 3    # all anchors are Gaussian
    #scenario = 2    # 1 anchor is exponential, 3 are Gaussian
    #scenario = 3    # all anchors are exponential
    
    # specify position of anchors
    p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
        #creates an array which has 4 rows and 2 cols; the rows are seperated by , (comma)
    nr_anchors = np.size(p_anchor,0) 
        #returns the size of the array along the 0-dimension (i.e. the amount of rows)
        #np.size(p_anchor,1) would return the amount of cols
       
    
    # position of the agent for the reference mearsurement
    p_ref = np.array([[0,0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2,-4]])
#    p_true = np.array([[2,-4])
                       
    plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)
        #given function creates a plot
    
    # load measured data and reference measurements for the chosen scenario
    data,reference_measurement = load_data(scenario)
        #unlike C, python can return 2 variables
        
    # get the number of measurements 
    
    assert(np.size(data,0) == np.size(reference_measurement,0))
 
    nr_samples = np.size(data,0)
    
    #1) ML estimation of model parameters
    #TODO 
    params = parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref,scenario)
    
    return 0
    
    #2) Position estimation using least squares
    #TODO
    position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, True)

    if(scenario == 3):
        # TODO: don't forget to plot joint-likelihood function for the first measurement

        #3) Postion estimation using numerical maximum likelihood
        #TODO
        position_estimation_numerical_ml(data,nr_anchors,p_anchor, params, p_true)
    
        #4) Position estimation with prior knowledge (we roughly know where to expect the agent)
        #TODO
        # specify the prior distribution
        prior_mean = p_true
        prior_cov = np.eye(2)
        position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov, params, p_true)

    pass
    

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref,scenario):
    """ estimate the model parameters for all 4 anchors based on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 2x2 """

    params = np.zeros([1, nr_anchors])
    #TODO (1) check whether a given anchor is Gaussian or exponential
    
    #p_sub = p_anchor - p_ref #get the distance to the reference point
    #reference_measurement = [ 
            #1 Messung:        [pa0, pa1, pa2, pa3]
            #2 Messung:        [pa0, pa1, pa2, pa3]
            # ...
            #N Messung:        [pa0, pa1, pa2, pa3] ]
            
            #pa0 - pa3 sind features
            
    #each feature has its own paramters
    N = np.size(reference_measurement,0) #N samples
    # nr_anchors is the amount of features
    
    
    if scenario == 1:
        #we know this
        distributions = ["Gaussian", "Gaussian", "Gaussian", "Gaussian"]
        
    if scenario == 2:
        fig, axs = plt.subplots(nr_anchors) #create a subplot with 4 horziontally spaced((1,4) would be vertically
        fig.suptitle('Histogramm of four measurements')
        
        for i in range(0,nr_anchors):  #range creates a vector from 0 to nr_anchors-1
            axs[i].hist(reference_measurement[:,i], color = 'blue', edgecolor = 'black', bins = 15)
            axs[i].set(ylabel='#Samples')
            
        # -1 is the last element
        axs[-1].set(xlabel='distance d')            
        plt.show()
        
        #seen from plot
        distributions = ["Exponential", "Gaussian", "Gaussian", "Gaussian"]
        
    if scenario == 3:
        #we know this
        distributions = ["Exponential", "Exponential", "Exponential", "Exponential"]        

   
        
    #TODO (2) estimate the according parameter based 

    # "for dist in distribution" would cycle thorugh all items in distribution, but 
    #there would be no integer counter inside the loop then.
    #With 2 parameters and enumerate() a counter can additonally be used w/o the use of an variabel like counter = 0; and inside: counter = counter + 1)
    for idx, dist in enumerate(distributions):  
        #print(list(enumerate(distributions))) #has to be a list in python 3 to display correctly
        if dist == "Gaussian":        
            #calculate the current mu with the given points of anchor and agent(refernce point) 
            mu = np.linalg.norm(p_anchor[idx,:] - p_ref)
            
            #calucalte sigma_sq with the formula from the lecture
            sigma_sq = np.sum((reference_measurement[:,idx] - mu)**2,0)/N 
            
            # sigma = np.sqrt(sigma)            
            params[0,idx] = sigma_sq
            
        elif dist == "Exponential":
            #calculate the current mu with the given points of anchor and agent(refernce point) 
            mu = np.linalg.norm(p_anchor[idx,:] - p_ref)
            
            #calculate lambda with the derived formula
            lambda_i = N/( np.sum(reference_measurement[:,idx]) - N*mu )
            #with lambda you can define an anonymus function, hence I had to rename it a little
            
            params[0,idx] = lambda_i
            
        else:
            print(dist, " distribution is not implemented")
    
    
    print(distributions)
    print(params)    
    
    return params
#--------------------------------------------------------------------------------
def position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, use_exponential):
    """estimate the position by using the least squares approximation. 
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        p_true... true position (needed to calculate error) 2x2 
        use_exponential... determines if the exponential anchor in scenario 2 is used, bool"""
    nr_samples = np.size(data,0)
    
    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter = ...  # maximum iterations for GN
    
    # TODO estimate position for  i in range(0, nr_samples)
    # least_squares_GN(p_anchor,p_start, measurements_n, max_iter, tol)
	# TODO calculate error measures and create plots----------------
    pass
#--------------------------------------------------------------------------------
def position_estimation_numerical_ml(data,nr_anchors,p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """
    #TODO
    pass
#--------------------------------------------------------------------------------
def position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov,lambdas, p_true):
    """ estimate the position by accounting for prior knowledge that is specified by a bivariate Gaussian
    Input:
         data...distance measurements to unkown agent, nr_measurements x nr_anchors
         nr_anchors... scalar
         p_anchor... position of anchors, nr_anchors x 2
         prior_mean... mean of the prior-distribution, 2x1
         prior_cov... covariance of the prior-dist, 2x2
         lambdas... estimated parameters (scenario 3), nr_anchors x 1
         p_true... true position (needed to calculate error), 2x2 """
    # TODO
    pass
#--------------------------------------------------------------------------------
def least_squares_GN(p_anchor,p_start, measurements_n, max_iter, tol):
    """ apply Gauss Newton to find the least squares solution
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_start... initial position, 2x1
        measurements_n... distance_estimate, nr_anchors x 1
        max_iter... maximum number of iterations, scalar
        tol... tolerance value to terminate, scalar"""
    # TODO
    pass
    
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,title="Title"):
    
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      title... title of the plot (optional), string"""
    
	#npts = 100
    delta = 0.025
    X, Y = np.mgrid[xmin:xmax:delta, ymin:ymax:delta]
    pos = np.dstack((X, Y))
                    
    Z = sp.stats.multivariate_normal(mu, cov)
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    plt.gca().set_aspect("equal")
    CS = plt.contour(X, Y, Z.pdf(pos),3,colors='r')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return

#--------------------------------------------------------------------------------
def ecdf(realizations):   
    """ computes the empirical cumulative distribution function for a given set of realizations.
    The output can be plotted by plt.plot(x,Fx)
    
    Input:
      realizations... vector with realizations, Nx1
    Output:
      x... x-axis, Nx1
      Fx...cumulative distribution for x, Nx1"""
    x = np.sort(realizations)
    Fx = np.linspace(0,1,len(realizations))
    return Fx,x

#--------------------------------------------------------------------------------
def load_data(scenario):
    """ loads the provided data for the specified scenario
    Input:
        scenario... scalar
    Output:
        data... contains the actual measurements, nr_measurements x nr_anchors
        reference.... contains the reference measurements, nr_measurements x nr_anchors"""
    data_file = 'measurements_' + str(scenario) + '.data'
    ref_file =  'reference_' + str(scenario) + '.data'
    
    data = np.loadtxt(data_file,skiprows = 0)
    reference = np.loadtxt(ref_file,skiprows = 0)
    
    return (data,reference)
#--------------------------------------------------------------------------------
def plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None):
    """ plots all anchors and agents
    Input:
        nr_anchors...scalar
        p_anchor...positions of anchors, nr_anchors x 2
        p_true... true position of the agent, 2x1
        p_ref(optional)... position for reference_measurements, 2x1"""
    # plot anchors and true position
    plt.axis([-6, 6, -6, 6])
    for i in range(0, nr_anchors):
        plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
        plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.show()
    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
