#Filename: HW1_skeleton.py

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy
from mpl_toolkits.mplot3d import Axes3D

showZeros = True
# True: shows all points int he plot;
# False: Shows only points hight than 0;
stepsize= 0.05


#Used for the CDU in one plot
global xForPrint
global fxForPrint
xForPrint = [[],[]]
fxForPrint  = [[],[]]

#--------------------------------------------------------------------------------
# Assignment 1
def main():

    # choose the scenario
    scenario = 3    # all anchors are Gaussian
    #scenario = 2    # 1 anchor is exponential, 3 are Gaussian
    #scenario = 3    # all anchors are exponential

    # specify position of anchors
    p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
    nr_anchors = np.size(p_anchor,0)

    # position of the agent for the reference mearsurement
    p_ref = np.array([[0,0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2,-4]])


    plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)

    # load measured data and reference measurements for the chosen scenario
    data,reference_measurement = load_data(scenario)

    # get the number of measurements
    assert(np.size(data,0) == np.size(reference_measurement,0))
    nr_samples = np.size(data,0)

    #1) ML estimation of model parameters
    #TODO
    params = parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref)

    #2) Position estimation using least squares
    #TODO
    position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, True)

    if(scenario == 3):

        print("\n3.3.1 Single Measurement")
        p_d0 = np.zeros(nr_anchors)
        help = np.zeros(nr_anchors)
        p_anchor_x  = p_anchor.T[0]
        p_anchor_y  = p_anchor.T[1]

        gridSize = int(10/stepsize)
        counter = np.zeros((gridSize, gridSize))

        Xplot, Yplot = np.mgrid[-5:5:stepsize, -5:5:stepsize]
        X = Xplot
        Y = Yplot

        help0 = np.zeros(Xplot.shape)
        p_Anchor0 = np.zeros(Xplot.shape)
        help1 = np.zeros(Xplot.shape)
        p_Anchor1 = np.zeros(Xplot.shape)
        help2 = np.zeros(Xplot.shape)
        p_Anchor2 = np.zeros(Xplot.shape)
        help3 = np.zeros(Xplot.shape)
        p_Anchor3 = np.zeros(Xplot.shape)

        help0 = data[0][0] - ((p_anchor_x[0]- X)**2 + (p_anchor_y[0]-Y)**2 )**0.5
        p_Anchor0 = params[0][0] * np.exp(- params[0][0] *  help0)
        p_Anchor0[np.where(help0 < 0)] = 0
        help1 = data[0][1] - ((p_anchor_x[1]- X)**2 + (p_anchor_y[1]-Y)**2 )**0.5
        p_Anchor1 = params[0][1] * np.exp(- params[0][1] *  help1)
        p_Anchor1[np.where(help1 < 0)] = 0
        help2 = data[0][2] - ((p_anchor_x[2]- X)**2 + (p_anchor_y[2]-Y)**2 )**0.5
        p_Anchor2 = params[0][2] * np.exp(- params[0][2] *  help2)
        p_Anchor2[np.where(help2 < 0)] = 0
        help3 = data[0][3] - ((p_anchor_x[3]- X)**2 + (p_anchor_y[3]-Y)**2 )**0.5
        p_Anchor3 = params[0][3] * np.exp(- params[0][3] *  help3)
        p_Anchor3[np.where(help3 < 0)] = 0

        p_d = p_Anchor0 * p_Anchor1 * p_Anchor2 * p_Anchor3

        arrayX, arrayY = np.where(p_d==p_d.max())
        counter[arrayX, arrayY] = counter[arrayX, arrayY] + 1

        print("Joint likelihood for the first Measurement")
        print("Maximum :",arrayX*stepsize-5,arrayY   *stepsize-5)

        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1, projection='3d')
        ax2 = fig.add_subplot(2,2,2, projection='3d')
        ax3 = fig.add_subplot(2,2,3, projection='3d')
        ax4 = fig.add_subplot(2,2,4, projection='3d')
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')

        if showZeros:

            ax1.scatter(Xplot,Yplot,p_Anchor0,alpha=0.6 , c="red")
            ax2.scatter(Xplot,Yplot,p_Anchor1,alpha=0.6 , c="green")
            ax3.scatter(Xplot,Yplot,p_Anchor2,alpha=0.6 , c="blue")
            ax4.scatter(Xplot,Yplot,p_Anchor3,alpha=0.6 , c="yellow")
            ax.scatter(Xplot,Yplot,p_d,  c="red")
            ax.plot(Xplot[arrayX,arrayY],Yplot[arrayX,arrayY], p_d.max(), 'bo', markersize= 10)

        else:
            Xplot = Xplot.flatten()
            Xplot = Xplot.tolist()
            Yplot = Yplot.flatten()
            Yplot = Yplot.tolist()
            counter = counter.flatten()
            counter = counter.tolist()

            p_Anchor0 = p_Anchor0.flatten()
            p_Anchor1 = p_Anchor1.flatten()
            p_Anchor2 = p_Anchor2.flatten()
            p_Anchor3 = p_Anchor3.flatten()
            p_d = p_d.flatten()
            p_d = p_d.tolist()
            p_Anchor0 = p_Anchor0.tolist()
            p_Anchor1 = p_Anchor1.tolist()
            p_Anchor2 = p_Anchor2.tolist()
            p_Anchor3 = p_Anchor3.tolist()

            for i in range(len(p_Anchor0)):
                if p_Anchor0[i] != 0:
                    ax1.plot([Xplot[i]],[Yplot[i]], p_Anchor0[i], 'ro', markersize= 10)
                if p_Anchor1[i] != 0:
                    ax2.plot([Xplot[i]],[Yplot[i]], p_Anchor1[i], 'bo', markersize= 10)
                if p_Anchor2[i] != 0:
                    ax3.plot([Xplot[i]],[Yplot[i]], p_Anchor2[i], 'go', markersize= 10)
                if p_Anchor3[i] != 0:
                    ax4.plot([Xplot[i]],[Yplot[i]], p_Anchor3[i], 'yo', markersize= 10)

            for i in range(len(counter)):
                if p_d[i] != 0:
                    ax.plot([Xplot[i]],[Yplot[i]], p_d[i], 'bo', markersize= 10)
        plt.show()

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
def parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref):
    """ estimate the model parameters for all 4 anchors based on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 2x2 """


    print("\n2. Maximum likelihood Estimation of Model Parameters")
    params = np.zeros([1, nr_anchors])

    anchor_0 = reference_measurement.T[0]
    anchor_1 = reference_measurement.T[1]
    anchor_2 = reference_measurement.T[2]
    anchor_3 = reference_measurement.T[3]


    # plt.hist(anchor_0, 50, density=True, facecolor='g')
    # plt.ylabel('# samples')
    # plt.xlabel('d / m')
    # plt.show()
    # plt.hist(anchor_1, 50, density=True, facecolor='g')
    # plt.ylabel('# samples')
    # plt.xlabel('d / m')
    # plt.show()
    # plt.hist(anchor_2, 50, density=True, facecolor='g')
    # plt.ylabel('# samples')
    # plt.xlabel('d / m')
    # plt.show()
    # plt.hist(anchor_3, 50, density=True, facecolor='g')
    # plt.ylabel('# samples')
    # plt.xlabel('d / m')
    # plt.show()

    # fig = plt.figure()
    # ax1 = fig.add_subplot(2,2,1)
    # ax2 = fig.add_subplot(2,2,2)
    # ax3 = fig.add_subplot(2,2,3)
    # ax4 = fig.add_subplot(2,2,4)
    # ax1.hist(anchor_0, 50, density=True, facecolor='g')
    # ax2.hist(anchor_1, 50, density=True, facecolor='g')
    # ax3.hist(anchor_2, 50, density=True, facecolor='g')
    # ax4.hist(anchor_3, 50, density=True, facecolor='g')
    # plt.show()

    mu_0 = ((0-5)**2 + (0-5)**2)**0.5
    mu_1 = ((0+5)**2 + (0-5)**2)**0.5
    mu_2 = ((0+5)**2 + (0+5)**2)**0.5
    mu_3 = ((0-5)**2 + (0+5)**2)**0.5

    ###################################Gaussian
    # mu_0 = np.sum(anchor_0)/ anchor_0.shape[0]
    # mu_1 = np.sum(anchor_1)/ anchor_1.shape[0]
    # mu_2 = np.sum(anchor_2)/ anchor_2.shape[0]
    # mu_3 = np.sum(anchor_3)/ anchor_3.shape[0]


    alpha = 1e-3
    k2_1, p_1 = stats.mstats.normaltest(anchor_0)
    k2_2, p_2 = stats.mstats.normaltest(anchor_1)
    k2_3, p_3 = stats.mstats.normaltest(anchor_2)
    k2_4, p_4 = stats.mstats.normaltest(anchor_3)

    if p_1 >=  alpha:
        var_0 = np.sum((anchor_0 - mu_0)**2) / anchor_0.shape[0]
        print("Anchor 0: Gaussian with sigma: ", var_0)
    else:
        var_0 = anchor_0.shape[0] / np.sum((anchor_0 - mu_0))
        print("Anchor 0: Expotnential with lambda: ", var_0)
    if p_2 >=  alpha:
        var_1 = np.sum((anchor_1 - mu_1)**2) / anchor_1.shape[0]
        print("Anchor 1: Gaussian with sigma: ", var_1)
    else:
        var_1 = anchor_1.shape[0] / np.sum((anchor_1 - mu_1))
        print("Anchor 1: Expotnential with lambda: ", var_1)
    if p_3 >=  alpha:
        var_2 = np.sum((anchor_2 - mu_2)**2) / anchor_2.shape[0]
        print("Anchor 2: Gaussian with sigma: ", var_2)
    else:
        var_2 = anchor_2.shape[0] / np.sum((anchor_2 - mu_2))
        print("Anchor 2: Expotnential with lambda: ", var_2)
    if p_4 >=  alpha:
        var_3 = np.sum((anchor_3 - mu_3)**2) / anchor_3.shape[0]
        print("Anchor 3: Gaussian with sigma: ", var_3)
    else:
        var_3 = anchor_3.shape[0] / np.sum((anchor_3 - mu_3))
        print("Anchor 3: Expotnential with lambda: ", var_3)

    params[0][0] = var_0
    params[0][1] = var_1
    params[0][2] = var_2
    params[0][3] = var_3

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

    print("\n3.2 Gauss-Newton Algorithm for Position Estimation")
    nr_samples = np.size(data,0)
    #TODO set parameters
    tol = 0.0001  # tolerance
    max_iter = 1000  # maximum iterations for GN

    xmin=-6
    ymin=-6
    xmax=6
    ymax=6

    point = np.zeros((data.shape[0],2))
    d_error = np.zeros(data.shape[0])

    if  not use_exponential:
        p_anchor = np.delete(p_anchor, 0,0)
        data =  np.delete(data.T, 0,0)
        data = data.T


    for i in range(0, nr_samples):

        p_start = [np.random.uniform(low=-5.0, high=5.0),np.random.uniform(low=-5.0, high=5.0)]
        measurements_n = data[i]
        point[i] = least_squares_GN(p_anchor,p_start, measurements_n, max_iter, tol)

    # TODO calculate error measures and create plots----------------

    d_error[:] =  ((point.T[0][:] - p_true[0][0])**2 +  (point.T[1][:] - p_true[0][1])**2)**0.5


    mu_error  = np.sum(d_error) / point.shape[0]
    var_error = np.sum((d_error - mu_error)**2) / point.shape[0]
    print("Error Mean: ",mu_error,"\nError Variance", var_error)

    mu_x = np.sum(point.T[0]) / point.shape[0]
    mu_y = np.sum(point.T[1]) / point.shape[0]
    cov = np.cov(point.T)
    plt.scatter(point.T[0], point.T[1])
    plt.xlim([xmin,xmax])
    plt.xlabel('x / m')
    plt.ylabel('y / m')
    plt.ylim([ymin,ymax])
    plot_gauss_contour([mu_x,mu_y], cov,xmin,xmax,ymin,ymax,title="Estimated Positons")
    Fx, x = ecdf(d_error)
    plt.plot(x,Fx)
    xForPrint[1] = x
    fxForPrint[1] = Fx
    plt.ylabel('P(point in d)')
    plt.xlabel('d / m')
    plt.show()
#--------------------------------------------------------------------------------
def position_estimation_numerical_ml(data,nr_anchors,p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """
    global xForPrint, fxForPrint
    print("\n3.3.2 Multiple Measurements")

    nr_samples = np.size(data,0)
    p_d0 = np.zeros(nr_anchors)
    help = np.zeros(nr_anchors)
    p_anchor_x  = p_anchor.T[0]
    p_anchor_y  = p_anchor.T[1]

    gridSize = int(10/stepsize)
    counter = np.zeros((gridSize, gridSize))
    Xplot, Yplot = np.mgrid[-5:5:stepsize, -5:5:stepsize]
    X = Xplot
    Y = Yplot

    help0 = np.zeros(Xplot.shape)
    p_Anchor0 = np.zeros(Xplot.shape)
    help1 = np.zeros(Xplot.shape)
    p_Anchor1 = np.zeros(Xplot.shape)
    help2 = np.zeros(Xplot.shape)
    p_Anchor2 = np.zeros(Xplot.shape)
    help3 = np.zeros(Xplot.shape)
    p_Anchor3 = np.zeros(Xplot.shape)
    d_error = []

    for k in range(nr_samples):

        help0 = data[k][0] - ((p_anchor_x[0]- X)**2 + (p_anchor_y[0]-Y)**2 )**0.5
        p_Anchor0 = lambdas[0][0] * np.exp(- lambdas[0][0] *  help0)
        p_Anchor0[np.where(help0 < 0)] = 0

        help1 = data[k][1] - ((p_anchor_x[1]- X)**2 + (p_anchor_y[1]-Y)**2 )**0.5
        p_Anchor1 = lambdas[0][1] * np.exp(- lambdas[0][1] *  help1)
        p_Anchor1[np.where(help1 < 0)] = 0

        help2 = data[k][2] - ((p_anchor_x[2]- X)**2 + (p_anchor_y[2]-Y)**2 )**0.5
        p_Anchor2 = lambdas[0][2] * np.exp(- lambdas[0][2] *  help2)
        p_Anchor2[np.where(help2 < 0)] = 0

        help3 = data[k][3] - ((p_anchor_x[3]- X)**2 + (p_anchor_y[3]-Y)**2 )**0.5
        p_Anchor3 = lambdas[0][3] * np.exp(- lambdas[0][3] *  help3)
        p_Anchor3[np.where(help3 < 0)] = 0

        p_d = p_Anchor0 * p_Anchor1 * p_Anchor2 * p_Anchor3

        arrayX, arrayY = np.where(p_d==p_d.max())

        counter[arrayX, arrayY] = counter[arrayX, arrayY] + 1
        d_error.append((((arrayX*stepsize-5) - p_true[0][0])**2 +  ((arrayY*stepsize-5) - p_true[0][1])**2)**0.5)

    bigX , bigY =    np.where(counter == counter.max())

    print("\nNumerical maximum likelihood based on the joint likelihood")
    print("Maximum :",bigX*stepsize-5,bigY*stepsize-5)

    d_error = np.asarray(d_error)
    d_error = d_error.flatten()
    mu_error  = np.sum(d_error) / d_error.shape[0]
    var_error = np.sum((d_error - mu_error)**2) / d_error.shape[0]
    print("Error Mean: ",mu_error,"\nError Variance", var_error)

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')

    if showZeros:
        ax.scatter(Xplot,Yplot,counter, c="red")
    else:
        Xplot = Xplot.flatten()
        Xplot = Xplot.tolist()
        Yplot = Yplot.flatten()
        Yplot = Yplot.tolist()

        counter = counter.flatten()
        counter = counter.tolist()

        for i in range(len(counter)):
            if counter[i] != 0:
                ax.plot([Xplot[i]],[Yplot[i]], counter[i], 'bo', markersize= 10)

    plt.xlim([-5,5])
    plt.xlabel('x / m')
    plt.ylabel('y / m')
    plt.ylim([-5,5])
    plt.show()

    Fx, x = ecdf(d_error)
    xForPrint[0] = x
    fxForPrint[0] = Fx
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
    global xForPrint, fxForPrint

    nr_samples = np.size(data,0)
    p_d0 = np.zeros(nr_anchors)
    help = np.zeros(nr_anchors)
    p_anchor_x  = p_anchor.T[0]
    p_anchor_y  = p_anchor.T[1]

    gridSize = int(10/stepsize)
    counter = np.zeros((gridSize, gridSize))
    Xplot, Yplot = np.mgrid[-5:5:stepsize, -5:5:stepsize]
    X = Xplot
    Y = Yplot

    help0 = np.zeros(Xplot.shape)
    p_Anchor0 = np.zeros(Xplot.shape)
    help1 = np.zeros(Xplot.shape)
    p_Anchor1 = np.zeros(Xplot.shape)
    help2 = np.zeros(Xplot.shape)
    p_Anchor2 = np.zeros(Xplot.shape)
    help3 = np.zeros(Xplot.shape)
    p_Anchor3 = np.zeros(Xplot.shape)
    d_error = []

    prior = (1/(2 * np.pi)) * np.exp(-(0.5 * ((X-2)**2 + (Y+4)**2)))

    for k in range(nr_samples):


        help0 = data[k][0] - ((p_anchor_x[0]- X)**2 + (p_anchor_y[0]-Y)**2 )**0.5
        p_Anchor0 = lambdas[0][0] * np.exp(- lambdas[0][0] *  help0) * prior
        p_Anchor0[np.where(help0 < 0)] = 0
        help1 = data[k][1] - ((p_anchor_x[1]- X)**2 + (p_anchor_y[1]-Y)**2 )**0.5
        p_Anchor1 = lambdas[0][1] * np.exp(- lambdas[0][1] *  help1) * prior
        p_Anchor1[np.where(help1 < 0)] = 0
        help2 = data[k][2] - ((p_anchor_x[2]- X)**2 + (p_anchor_y[2]-Y)**2 )**0.5
        p_Anchor2 = lambdas[0][2] * np.exp(- lambdas[0][2] *  help2) * prior
        p_Anchor2[np.where(help2 < 0)] = 0
        help3 = data[k][3] - ((p_anchor_x[3]- X)**2 + (p_anchor_y[3]-Y)**2 )**0.5
        p_Anchor3 = lambdas[0][3] * np.exp(- lambdas[0][3] *  help3) * prior
        p_Anchor3[np.where(help3 < 0)] = 0
        p_d = p_Anchor0 * p_Anchor1 * p_Anchor2 * p_Anchor3
        arrayX, arrayY = np.where(p_d==p_d.max())
        counter[arrayX, arrayY] = counter[arrayX, arrayY] + 1
        d_error.append((((arrayX*stepsize-5) - p_true[0][0])**2 +  ((arrayY*stepsize-5) - p_true[0][1])**2)**0.5)


    bigX , bigY =    np.where(counter == counter.max())

    print("\nBayesian estimator")
    print("Maximum :",bigX*stepsize-5,bigY*stepsize-5)

    d_error = np.asarray(d_error)
    d_error = d_error.flatten()

    mu_error  = np.sum(d_error) / d_error.shape[0]
    var_error = np.sum((d_error - mu_error)**2) / d_error.shape[0]
    print("Error Mean: ",mu_error,"\nError Variance", var_error)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    if showZeros:
        ax.scatter(Xplot,Yplot,counter, c="red")
    else:
        Xplot = np.delete(Xplot, np.where(counter==0))
        Yplot = np.delete(Yplot, np.where(counter==0))

        Xplot = Xplot.flatten()
        Xplot = Xplot.tolist()
        Yplot = Yplot.flatten()
        Yplot = Yplot.tolist()

        counter = counter.flatten()
        counter = counter.tolist()

        for i in range(len(counter)):
            if counter[i] != 0:
                ax.plot([Xplot[i]],[Yplot[i]], counter[i], 'bo', markersize= 10)

    plt.xlim([-5,5])
    plt.xlabel('x / m')
    plt.ylabel('y / m')
    plt.ylim([-5,5])
    plt.show()

    Fx, x = ecdf(d_error)
    plt.plot(x,Fx)
    plt.plot(xForPrint[0],fxForPrint[0])
    plt.plot(xForPrint[1],fxForPrint[1])
    plt.legend(["bayesian","numerical maximum likelihood","least-squares"])
    plt.ylabel('P(point in d)')
    plt.xlabel('d / m')
    plt.show()
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
    nr_anchors = p_anchor.shape[0]
    jacobi = np.zeros((nr_anchors,2))

    p_anchor_x  = p_anchor.T[0]
    p_anchor_y  = p_anchor.T[1]

    d = np.zeros(nr_anchors)
    d_n = measurements_n
    p_now = p_start

    for i in range(max_iter):

        for j in range(nr_anchors):
            d[j] = (((p_anchor_x[j]-p_now[0])**2 + (p_anchor_y[j]-p_now[1])**2 )**0.5)
            jacobi[j][0] = (p_anchor_x[j]-p_now[0]) / (((p_anchor_x[j]-p_now[0])**2 + (p_anchor_y[j]-p_now[1])**2 )**0.5)
            jacobi[j][1] = (p_anchor_y[j]-p_now[1]) / (((p_anchor_x[j]-p_now[0])**2 + (p_anchor_y[j]-p_now[1])**2 )**0.5)

        help = np.matmul(np.linalg.inv(np.matmul(jacobi.T,jacobi)), np.matmul(jacobi.T, (d_n - d)))
        p_t1 = p_now - help

        d_now = ((p_t1[0] -p_now[0])**2 + (p_t1[1]-p_now[1])**2)**0.5

        if d_now < tol:
            return p_t1

        p_now = p_t1

    return None

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

    Z = stats.multivariate_normal(mu, cov)
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
