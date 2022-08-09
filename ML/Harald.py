# Filename: HW1_skeleton.py

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy
from mpl_toolkits.mplot3d import axes3d, Axes3D
# https://stackoverflow.com/questions/3810865/matplotlib-unknown-projection-3d-error
# because I got this error in Jupyter Notebook for 3D plots before
import sys  # required for exception handling, used np.inv instead of np.linalg.inv


# --------------------------------------------------------------------------------
# Assignment 1
def main():
    # choose the scenario
    #scenario = 1    # all anchors are Gaussian
    #scenario = 2    # 1 anchor is exponential, 3 are Gaussian
    use_exp_anchor = False  # task 3.2: neglect the anchor with exp. distr., this param affects only scenario 2
    scenario = 3  # all anchors are exponential

    # sanity check
    if scenario != 2:
        use_exp_anchor = True

    # specify position of anchors
    p_anchor = np.array([[5, 5], [-5, 5], [-5, -5], [5, -5]])
    nr_anchors = np.size(p_anchor, 0)

    # position of the agent for the reference mearsurement
    p_ref = np.array([[0, 0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2, -4]])
    #    p_true = np.array([[2,-4])

    plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)

    # load measured data and reference measurements for the chosen scenario
    data, reference_measurement = load_data(scenario)

    # get the number of measurements 
    assert (np.size(data, 0) == np.size(reference_measurement, 0))
    nr_samples = np.size(data, 0)

    # 1) ML estimation of model parameters
    # TODO
    params = parameter_estimation(reference_measurement, nr_anchors, p_anchor, p_ref)

    # 2) Position estimation using least squares
    # TODO
    # param 'use_exp_anchor', please take a look above to switch on/off
    position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, use_exp_anchor)

    if (scenario == 3):
        # TODO: don't forget to plot joint-likelihood function for the first measurement
        min_anchor0 = np.min(data[:,0])
        min_anchor1 = np.min(data[:,1])
        min_anchor2 = np.min(data[:,2])
        min_anchor3 = np.min(data[:,3])
        print("Min distances: ",min_anchor0,",",min_anchor1,",",min_anchor2,",",min_anchor3)
        measurement_0 = data[0]
        xmin = np.min(p_anchor[:, 0])
        xmax = np.max(p_anchor[:, 0])
        ymin = np.min(p_anchor[:, 1])
        ymax = np.max(p_anchor[:, 1])
        delta = 0.05

        fig = plt.figure()

        #ax = fig.gca(projection='3d') #causes an exception in Jupyter therefore look at next line
        ax = Axes3D(fig)
        # Make data.
        X = np.arange(xmin, xmax, delta)
        Y = np.arange(ymin, ymax, delta)
        X, Y = np.meshgrid(X, Y)
        joint_likelihoods=[]
        for anchor_index in range(0, nr_anchors):
            anchor = p_anchor[anchor_index]
            #lambda_anchor = params[0, anchor_index]
            #lambda_anchor = 0.3#debug
            measurement_0_to_anchor = measurement_0[anchor_index]
            #true_diff = anchor - p_true
            #true_distance = np.linalg.norm(true_diff)  # distance from p_true to anchor
            #print("p_true to anchor ",anchor_index,": ",true_distance,", measure=",measurement_0[anchor_index])
            anchor_x = anchor[0]
            anchor_y = anchor[1]
            dbg1 = (X - anchor_x) ** 2
            dbg2 = (Y - anchor_y) ** 2
            dbg3 = dbg1 + dbg2
            grid_of_distances_to_anchor = np.sqrt(dbg3)  # contains distance of each grid point to anchor
            #Likelihood is computed by inserting ML solution for lambda into exponential distribution formula
            #likelihood for one single measurement: 1.0/(measurement-distance(point,anchor))*exp(-1.0)
            #this comes from ML solution for lambda for N=1 (single measurement)
            grid_of_distances_minus_measured_dist = np.subtract(grid_of_distances_to_anchor, measurement_0_to_anchor)
            #elements bigger than measurement have to be infinite, subsequent 1.0/inf=0 for likelihood
            grid_of_distances_minus_measured_dist[grid_of_distances_minus_measured_dist >= 0] = float("inf")
            #compute 1.0/.. for each element
            likelihood_for_anchor_i = np.reciprocal(grid_of_distances_minus_measured_dist)
            #final multiplication by 1.0/exp(1)
            likelihood_for_anchor_i = np.multiply(likelihood_for_anchor_i,np.exp(-1.0))
            joint_likelihoods.append(likelihood_for_anchor_i)
            # plt.close('all')
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # surf = ax.plot_surface(X, Y, likelihood_for_anchor_i, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # plt.title("Debug Likelihood for anchor "+str(anchor_index))
            # plt.show()
        joint_likelihood=joint_likelihoods[0]
        for anchor_index in range(1,nr_anchors):
            joint_likelihood=np.multiply(joint_likelihood,joint_likelihoods[anchor_index])

        plt.close('all')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, joint_likelihood, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        print("Joint Likelihood")
        print("Q: Why might it be hard to ﬁnd the maximum of this function with a gradient ascent algorithm using an arbitrary starting point within the evaluation region?")
        print("A: Most of the floor's area is flat, no slope in any direction. Gradient Ascending Algorithm will wander around due to numerical")
        print("fluctuations.")
        print("Additionally there are multiple local maxima, which may catch a gradient ascending algorithm.")
        print("\nQ:Is the maximum at the true position?")
        print("A:No, because position estimations are based on 'distorted' measurements. Measurements ")
        print("will be close to the true position with high probability, but they deviate from the true position. If measurements were exact,")
        print("position derivation would be an analytical task and three measurements would suffice to derive true position.")
        plt.title("Joint Likelihood for first measurement")

        plt.show()

        # 3) Postion estimation using numerical maximum likelihood
        # TODO
        position_estimation_numerical_ml(data, nr_anchors, p_anchor, params, p_true)

        # 4) Position estimation with prior knowledge (we roughly know where to expect the agent)
        # TODO
        # specify the prior distribution
        prior_mean = p_true
        prior_cov = np.eye(2)
        position_estimation_bayes(data, nr_anchors, p_anchor, prior_mean, prior_cov, params, p_true)

    pass


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def parameter_estimation(reference_measurement, nr_anchors, p_anchor, p_ref):
    """ estimate the model parameters for all 4 anchors based on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 2x2 """
    params = np.zeros([1, nr_anchors])
    # gaussian estimation
    gaussian_std = params
    for anchor_index in range(0, nr_anchors):
        anchor_coordinates = p_anchor[anchor_index]
        x_reference_distance = anchor_coordinates[0] - p_ref[0, 0]  # get x coordinate distance
        y_reference_distance = anchor_coordinates[1] - p_ref[0, 1]  # get y coordinate distance
        true_anchor_reference_distance = np.sqrt(x_reference_distance ** 2 + y_reference_distance ** 2)
        reference_measurements_for_anchor_i = reference_measurement[:, anchor_index]
        # TODO (1) check whether a given anchor is Gaussian or exponential
        is_exponential = True
        # Within an exponential distribution all values must be bigger than true distance length (true_anchor_reference_distance)
        # Therefore check if there is a value less than true distance. If found, this can not be an exponential distribution
        number_of_reference_measurements = reference_measurements_for_anchor_i.shape[0]
        for reference_measure_index in range(0, number_of_reference_measurements):
            ref_meas = reference_measurements_for_anchor_i[reference_measure_index]
            if ref_meas < true_anchor_reference_distance:
                is_exponential = False
                break
        # TODO (2) estimate the according parameter based
        if is_exponential:
            # estimate lambda
            ref_meas_dev_sum = 0
            for reference_measure_index in range(0, number_of_reference_measurements):
                ref_meas = reference_measurements_for_anchor_i[reference_measure_index]
                ref_meas_dev_sum += (ref_meas - true_anchor_reference_distance)
            exp_distribution_lambda = number_of_reference_measurements / ref_meas_dev_sum
            params[0, anchor_index] = exp_distribution_lambda
            print("Anchor " + str(anchor_index) + " EXP: lambda=", exp_distribution_lambda)
        else:
            # gaussian distribution, estimate standard deviation, do not use np.std because it uses the mean from the population (but mean of true value should be used)
            ref_meas_dev_sum = 0
            for reference_measure_index in range(0, number_of_reference_measurements):
                ref_meas = reference_measurements_for_anchor_i[reference_measure_index]
                dbg_diff = ref_meas - true_anchor_reference_distance
                dbg_sqr = (dbg_diff) ** 2
                ref_meas_dev_sum += dbg_sqr
            ref_meas_dev_sum = ref_meas_dev_sum / number_of_reference_measurements
            anchor_std = np.sqrt(ref_meas_dev_sum)
            params[0, anchor_index] = anchor_std
            print("Anchor " + str(anchor_index) + " GAUSS: std=", anchor_std)
    return params


# --------------------------------------------------------------------------------
def position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, use_exponential):
    """estimate the position by using the least squares approximation. 
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        p_true... true position (needed to calculate error) 2x2 
        use_exponential... determines if the exponential anchor in scenario 2 is used, bool"""
    nr_samples = np.size(data, 0)

    # TODO set parameters
    tol = 0.01  # ...  # tolerance
    max_iter = 50  # ...  # maximum iterations for GN
    # most of the time toleration is underrun after about 5-10 iterations, so max_iter is a very pessimistic guess

    if (use_exponential == False):
        # from analysis before I know that anchor 0 is exponentially distributed in scenario 2
        # 3.2, "Consider scenario 2" I remove anchor 0's measurements in order to keep following code backwards compatible
        p_anchor = np.delete(p_anchor, 0, 0)  # remove first row (=anchor 0)
        data = np.delete(data, 0, 1)  # remove first column (=anchor 0)
        nr_anchors = nr_anchors - 1

    # find bounding box of anchor positions for taking a random start point out of spanned area (uniform distribution)
    min_x = np.min(p_anchor[:, 0])
    max_x = np.max(p_anchor[:, 0])
    min_y = np.min(p_anchor[:, 1])
    max_y = np.max(p_anchor[:, 1])

    # TODO estimate position for  i in range(0, nr_samples)
    prepare_plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None)
    # prepare variables
    estimated_positions = np.zeros((nr_samples, 2))
    estimated_errors = np.zeros(nr_samples)
    for i in np.arange(0, nr_samples):
        measurements_n = data[i]
        p_start_x = np.random.uniform(min_x, max_x)
        p_start_y = np.random.uniform(min_y, max_y)
        p_start = np.array([[p_start_x, p_start_y]])
        # p_start = np.array([[0.0, 0.0]])#easier debugging
        estimated_pos = least_squares_GN(p_anchor, p_start, measurements_n, max_iter, tol)
        estimated_positions[i][0] = estimated_pos[0][0]  # x coord
        estimated_positions[i][1] = estimated_pos[0][1]  # y coord
        # compute estimation error as difference between estimated pos and true position "p_true"
        estimated_diff_from_p_true = estimated_pos - p_true
        estimated_dist_from_p_true = np.linalg.norm(estimated_diff_from_p_true)
        estimated_errors[i] = estimated_dist_from_p_true
        # Scatter plots of the estimated positions...
        plt.scatter(estimated_pos[0, 0], estimated_pos[0, 1], c='green', marker='x')
        # TODO calculate error measures and create plots----------------
    # plt.show()
    # last plot ensures visibility by overwriting old ones
    plt.plot(p_true[0, 0], p_true[0, 1], 'bo')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, '$p_{true}$')
    # The mean and variance of the position estimation error ...
    error_avg = np.average(estimated_errors)
    error_std = np.std(estimated_errors)
    print("3.2. estimated errors mean=", error_avg, ",std=", error_std)

    # Fit a two-dimensional Gaussian distribution to the point cloud of estimated positions and draw its contour lines ...
    mu = []
    mu.append(np.average(estimated_positions[:, 0]))
    mu.append(np.average(estimated_positions[:, 1]))
    estimated_positions_trans = np.transpose(estimated_positions)
    cov = np.cov(estimated_positions_trans)
    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, "Estimated Positions' Distribution")

    # Do the estimated positions look Gaussian?
    print("Q: Do the estimated positions look Gaussian?")
    print(
        "A: Distribution of estimated positions follows distribution characteristics of measured distances. If they are gaussian,")
    print(
        "estimated positions are gaussian too (scenario 1). Nice to see at scenario 2: Anchor 0 is exp. distributed -> Distribution of")
    print(
        "estimated positions towards anchor 0 is exponentially distributed too, whereas all other directions are gaussian.")
    print("Fitting of a gaussian for exponentially distributed positions, moves the gaussian center (mean)")
    print("away from the 'true' point.")

    # Compare the diﬀerent scenarios by looking at the cumulative distribution function (CDF) of the position estimation error...
    print(
        "\nQ: What can you say about the probability of large estimation errors, and how does this depend on the scenario?")
    print("A: Scenario 1. ECDF reaches 1.0 at about 0.6, indicating a narrow gaussian bell curve. Low probability ")
    print("for large estimation errors.")
    print(
        "A: Scenario 2. ECDF reaches 1.0 at about 2,5~3, indicating a broader band for estimation errors. Higher probability")
    print(" for large estimation errors than in scenario 1 (this comes from exp. distr. of anchor 0).")
    print(
        "A: Scenario 3. ECDF reaches 1.0 at about 4~5, indicating the broades band for estimation errors of all scenarios here.")
    print("High probability for large estimation errors.")

    print("\nQ: Consider scenario 2: ...Therefore, neglect the anchor with the exponentially... can you observe?")
    print(
        "A:Everything is fine! All distributions are gaussian, all assumptions based on being gaussian are true, results are consistent.")
    print(
        "Estimated positions are gaussian distributed, fitting of gaussian contour is perfect... because disturbing exp. distribution is")
    print(
        "missing now. ECDF is also not distracted by exp. distribution, ECDF shows underlying gaussian distributions, probability of")
    print(
        "large errors is consistent with underlying gaussian distributions (now smaller than before with exp. distrib.)")
    print(
        "ECDF shows larger probability for larger errors than in Scenario 1, because position has been estimated based on only three")
    print(
        "distances instead of four. Therefore the estimation is less precise than in scenario 1 when using four distances for position estimation.")
    Fx, x = ecdf(estimated_errors)
    plt.plot(x, Fx)
    plt.show()
    pass


# --------------------------------------------------------------------------------
def position_estimation_numerical_ml(data, nr_anchors, p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """
    # TODO
    xmin = np.min(p_anchor[:, 0])
    xmax = np.max(p_anchor[:, 0])
    ymin = np.min(p_anchor[:, 1])
    ymax = np.max(p_anchor[:, 1])
    delta = 0.05

    # Make data.
    X = np.arange(xmin, xmax, delta)
    Y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(X, Y)
    joint_likelihoods = []
    nr_measurements = len(data)
    #nr_measurements = 5#dbg
    print("\nPlease wait, Joint Likelihood for N measurements is computed...")
    for measurement_index in range(0, nr_measurements):
        measurement = data[measurement_index]
        for anchor_index in range(0, nr_anchors):
            anchor = p_anchor[anchor_index]
            # lambda_anchor = params[0, anchor_index]
            # lambda_anchor = 0.3#debug
            measurement_to_anchor = measurement[anchor_index]
            # true_diff = anchor - p_true
            # true_distance = np.linalg.norm(true_diff)  # distance from p_true to anchor
            # print("p_true to anchor ",anchor_index,": ",true_distance,", measure=",measurement_0[anchor_index])
            anchor_x = anchor[0]
            anchor_y = anchor[1]
            dbg1 = (X - anchor_x) ** 2
            dbg2 = (Y - anchor_y) ** 2
            dbg3 = dbg1 + dbg2
            grid_of_distances_to_anchor = np.sqrt(dbg3)  # contains distance of each grid point to anchor
            # Likelihood is computed by inserting ML solution for lambda into exponential distribution formula
            # likelihood for one single measurement: 1.0/(measurement-distance(point,anchor))*exp(-1.0)
            # this comes from ML solution for lambda for N=1 (single measurement)
            grid_of_distances_minus_measured_dist = np.subtract(grid_of_distances_to_anchor, measurement_to_anchor)
            # elements bigger than measurement have to be infinite, subsequent 1.0/inf=0 for likelihood
            grid_of_distances_minus_measured_dist[grid_of_distances_minus_measured_dist >= 0] = float("inf")
            # compute 1.0/.. for each element
            likelihood_for_anchor_i = np.reciprocal(grid_of_distances_minus_measured_dist)
            likelihood_for_anchor_i = np.multiply(likelihood_for_anchor_i, np.exp(-1.0))
            joint_likelihoods.append(likelihood_for_anchor_i)
    joint_likelihood = joint_likelihoods[0]
    for likelihood_index in range(1, nr_measurements):
        joint_likelihood = np.multiply(joint_likelihood, joint_likelihoods[likelihood_index])

    plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection='3d') #causes an exception in Jupyter therefore look at next line
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, joint_likelihood, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title("Joint Likelihood for N measurements")

    print("Q: Compare the performance of this estimator...to the least-squares algorithm for the data from scenario 3.")
    print("A: In Scenario 3 all anchors' measurements obey exponential distributions. Exponential distributions have")
    print("their highest probability at their 'true' distance (x=0 in literature) and decline over the distance. Least squares method tries to ")
    print("minimize deviation from anchors' measured distances. Distribution/Probability comes into play by frequency of measured distances but")
    print("least squares method does not consider them. ")
    print("Taking the average (x/y positions) of all random variable results (position estimations) will result in some distance")
    print("of this 'true' point. Position estimations also have limited accuracy due to tolerance values and maximum number of iterations,")
    print("which add additional errors to the least squares error method.")
    print("ML method does not require taking an average and therefore does not suffer from those limitations. ML is able to model")
    print("position estimations for exponential distributions too. It does not move away the most likely position from the 'true' point")
    print("like when taking the average of all estimations. Therefore ML is superior to least squares in this example (non guassian).")
    print("\nQ: Is this comparison fair? Is this truly a maximum likelihood estimator?")
    print("A: Computing a joined likelihood for each measurement (to all anchors) and multiplying them together results in a different")
    print("formula than for computing the maximum likelihood by considering all measurements at once. This can be analysed")
    print("for lambdas: They are simply multiplied together instead of using the ML solution for exponential distribution.")
    print("Using ML solution for N=1 also results in exp(-1), in contrast")
    print("to the ML solution for lambda, when considering all measurements at once.")

    plt.show()
    pass


# --------------------------------------------------------------------------------
def position_estimation_bayes(data, nr_anchors, p_anchor, prior_mean, prior_cov, lambdas, p_true):
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


# --------------------------------------------------------------------------------
def least_squares_GN(p_anchor, p_start, measurements_n, max_iter, tol):
    """ apply Gauss Newton to find the least squares solution
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_start... initial position, 2x1
        measurements_n... distance_estimate, nr_anchors x 1
        max_iter... maximum number of iterations, scalar
        tol... tolerance value to terminate, scalar"""
    # TODO
    anchor_count = p_anchor.shape[0]  # -2#verify implementation: crossings of 2 circles must be least squares solution
    J = np.empty([anchor_count, 2])
    p = p_start
    d_p = np.empty([anchor_count])
    loop_counter = 0
    # prepare_plot_anchors_and_agent(anchor_count, p_anchor, True, p_start)
    # plt.plot(p_start[0, 0], p_start[0, 1], 'bx')
    p_true = np.array([[2, -4]])
    # plt.plot(p_true[0, 0], p_true[0, 1], 'g*')
    measures = measurements_n[0:anchor_count]
    while True:
        for anchor_index in np.arange(0, anchor_count):
            anchor = p_anchor[anchor_index]
            diff_anchor_p = anchor - p
            dist = np.linalg.norm(diff_anchor_p)  # distance(anchor_x,anchor_y,p_x,p_y)
            partial_derivatives = 1.0 / dist * diff_anchor_p
            J[anchor_index] = partial_derivatives
            d_p[anchor_index] = dist
        # do it step by step
        J_trans = np.transpose(J)
        J_pseudo = np.dot(J_trans, J)
        J_pseudo_inv = np.linalg.inv(J_pseudo)
        dbg1 = np.dot(J_pseudo_inv, J_trans)
        dbg2 = measures - d_p
        dbg3 = np.dot(dbg1, dbg2)
        old_p = copy.copy(p)  # force deep copy
        p[0, 0] = p[0, 0] - dbg3[0]
        p[0, 1] = p[0, 1] - dbg3[1]
        # plt.plot(p[0, 0], p[0, 1], 'r*')
        loop_counter = loop_counter + 1
        if loop_counter >= max_iter:
            break
        diff_old_p_estim_p = old_p - p
        distance_gone = np.linalg.norm(diff_old_p_estim_p)

        if distance_gone < tol:
            # print("m=",measurements_n[0],",",measurements_n[1],",",measurements_n[2],",",measurements_n[3])
            # for anchor_index in np.arange(0, anchor_count):
            #     anchor = p_anchor[anchor_index]
            #     anchor_x = anchor[0]
            #     anchor_y = anchor[1]
            #     p_x = p[0, 0]
            #     p_y = p[0, 1]
            #     dist = distance(anchor_x, anchor_y, p_x, p_y)
            #     print("d",anchor_index,"=",dist)
            #     fig = plt.gcf()
            #     ax = fig.gca()
            #     circle = plt.Circle((anchor_x, anchor_y), measurements_n[anchor_index], color='b', fill=False, clip_on=True)
            #     ax.add_artist(circle)
            #print("Distance gone: ",distance_gone,", loops: ", loop_counter,", last p=",p[0,0],",",p[0,1])
            break
    # plt.show()
    return p
    # pass


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------
def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, title="Title"):
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      title... title of the plot (optional), string"""

    # npts = 100

    delta = 0.025
    X, Y = np.mgrid[xmin:xmax:delta, ymin:ymax:delta]
    pos = np.dstack((X, Y))

    Z = stats.multivariate_normal(mu, cov)
    plt.plot([mu[0]], [mu[1]], 'r+')  # plot the mean as a single point
    plt.gca().set_aspect("equal")
    CS = plt.contour(X, Y, Z.pdf(pos), 3, colors='r')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return


# --------------------------------------------------------------------------------
def ecdf(realizations):
    """ computes the empirical cumulative distribution function for a given set of realizations.
    The output can be plotted by plt.plot(x,Fx)
    
    Input:
      realizations... vector with realizations, Nx1
    Output:
      x... x-axis, Nx1
      Fx...cumulative distribution for x, Nx1"""
    # x = np.sort(realizations)#original code
    x = np.sort(realizations, axis=None)
    Fx = np.linspace(0, 1, len(realizations))
    return Fx, x


# --------------------------------------------------------------------------------
def load_data(scenario):
    """ loads the provided data for the specified scenario
    Input:
        scenario... scalar
    Output:
        data... contains the actual measurements, nr_measurements x nr_anchors
        reference.... contains the reference measurements, nr_measurements x nr_anchors"""
    data_file = 'measurements_' + str(scenario) + '.data'
    ref_file = 'reference_' + str(scenario) + '.data'

    data = np.loadtxt(data_file, skiprows=0)
    reference = np.loadtxt(ref_file, skiprows=0)

    return (data, reference)


# --------------------------------------------------------------------------------
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


def prepare_plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None):
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
    # plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    # plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    # plt.show()
    pass


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
