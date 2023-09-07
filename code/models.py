# imports go here
import numpy as np


class LinearRegressionModel:
    def __init__(self, locations=None, lengthscale=1.0, gamma=0.0):
        """
        Class constructor
        :param locations (optional): 2-by-m numpy array of RBF location, each col is coords of basis location
        :param lengthscale (default:1.0): float, lengthscale for radial basis functions
        :param gamma (default:0.0): float, regularisation parameter
        :return object of class LinearRegressionModel
        """
        self.locations = locations              
        self.lengthscale = lengthscale          
        if locations is not None:
            self.m = locations.shape[1]
        else:
            self.m = None
        self.gamma = gamma         
        self.theta1 = None
        self.theta2 = None
        self.theta3 = None

    def set_locations(self, locations):
        """
        Sets the locations attribute
        :param locations: 2-by-m numpy array of RBF location, each col is coords of basis location
        :return: None
        """
        self.locations = locations 
        self.m = locations.shape[1]

    def set_lengthscale(self, lengthscale):
        """
        Sets the lengthscale attribute
        :param lengthscale: float, lengthscale for RBFs
        :return: None
        """
        self.lengthscale = lengthscale

    def set_gamma(self, gamma):
        """
        Sets the gamma attribute
        :param gamma: float, regularisation parameter
        :return: None
        """
        self.gamma = gamma

    def set_parameters(self, theta1, theta2, theta3):
        """
        Allows user to manually set the model paraemters theta1, theta2, theta3
        :param theta1: numpy array of shape (m,) or (m, 1), parameters for model of y1
        :param theta2: numpy array of shape (m,) or (m, 1), parameters for model of y2
        :param theta3: numpy array of shape (m,) or (m, 1), parameters for model of y3
        :return: None
        """
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def save_params(self, file_name):
        """
        Saves the parameters theta1, theta2, and theta3 to the specified file
        :param file_name: string, e.g. "model_parameters.npz"
        :return: None
        """
        np.savez(file_name,theta1=self.theta1,theta2=self.theta2,theta3=self.theta3)


    def load_params(self,file_name):
        """
        Loads the parameters theta1, theta2, theta3
        :param file_name: string, e.g. "model_parameters.npy"
        :return: None
        """
        params = np.load(file_name, allow_pickle=True)
        self.theta1 = params['theta1']
        self.theta2 = params['theta2']
        self.theta3 = params['theta3']

    def build_phi_matrix(self, X):
        """
        Builds the phi matrix given a set of inputs
        :param X: (2,N) input array where N is the number of inputs and each col contains the coordinates of location
        :return phi: (N,m) numpy array, the regressor matrix
        """
        locations = self.locations
        lengthscale = self.lengthscale
        # calculate the regressor matrix phi
        # YOUR CODE HERE
        X1 = X[0, :]
        X2 = X[1, :]
        L1 = locations[0, :]
        L2 = locations[1, :]
        import numpy as np
        N = len(X[1,:])
        m = int(self.m)
        phi = np.zeros((N, m))

        # this method was foudn using ChatGPT
        # Prompt: achieve this as fast as possible:         for i in range(N):
        #             for k in range(m):
        #                 d = np.sqrt((X1[i] - L1[k])**2 + (X2[i] - L2[k])**2)
        #                 phi[i, k] = np.exp(-(1/(2 * lengthscale**2)) * (d)**2)
        X1 = X1[:, np.newaxis]  # Shape: (N, 1)
        X2 = X2[:, np.newaxis]  # Shape: (N, 1)
        L1 = L1[np.newaxis, :]  # Shape: (1, m)
        L2 = L2[np.newaxis, :]  # Shape: (1, m)

        # Compute the distances
        d = np.sqrt((X1 - L1) ** 2 + (X2 - L2) ** 2)

        # Compute phi
        phi = np.exp(-(1 / (2 * lengthscale ** 2)) * d ** 2)
        return phi


    def predict(self, X):
        """
        Calculates predictions of the outputs at supplied input points X
        :param X: (2,N) input array where N is the number of inputs and each col contains the coordinates of location
        :return y1hat, y2hat, y3hat: a tuple of 3 numpy arrays of shape (N,) or (N,1)
        """
        # should call the build_phi_matrix method
        # and then use the parameters self.theta1, self.theta2, self.theta3
        # and the regressor matrix to predict the output values at X
        theta1 = self.theta1
        theta2 = self.theta2
        theta3 = self.theta3
        # YOUR CODE HERE
        phi = self.build_phi_matrix(X)


        # calculate the predicted outputs
        y1hat = phi @ theta1
        y2hat = phi @ theta2
        y3hat = phi @ theta3
        return y1hat, y2hat, y3hat

    def fit(self, X, Y):
        """
        Determines the parameters for your model that best fit the training data (X,Y)
        :param X: (2,N) input array where N is the number of inputs/measurements
        and each col contains the values for [x1, x2]
        :param Y: (3,N) output array, where N is the number of inputs/measurements
        and each column contains the values [y1, y2, y3] at the location [x1, x2]
        :return: None
        """

        # Build regressor matrix
        # use regularised least squares to fit the models
        # and determine theta1, theta2, and theta3
        # YOUR CODE HERE
        phi = self.build_phi_matrix(X)
        gamma = self.gamma
        I = np.identity(self.m)

        self.theta1 = np.linalg.inv(phi.T @ phi + gamma * I) @ phi.T @ Y[0, :]
        self.theta2 = np.linalg.inv(phi.T @ phi + gamma * I) @ (phi.T @ Y[1, :])
        self.theta3 = np.linalg.inv(phi.T @ phi + gamma * I) @ (phi.T @ Y[2, :])

    def load_trained_model(self):
        N_mesh_x = np.linspace(0, 1.6, 41)
        N_mesh_y = np.linspace(0, 2.1, 41)

        xlocation, ylocation = np.meshgrid(N_mesh_x, N_mesh_y)
        xlocation = np.reshape(xlocation, (1, 41 * 41))
        xlocation -= 0.1
        ylocation = -ylocation - 1.7
        ylocation = np.reshape(ylocation, (1, 41 * 41))
        locations = np.vstack((xlocation, ylocation)) # ?? the radial basis function locations you used for your final trained model model
        lengthcale = 0.13420781 # ?? the lengthscale you used for your final trained model
        gamma = 0.04641589 # ?? the regularisation term you used for your final trained model
        self.set_locations(locations)
        self.set_lengthscale(lengthcale)
        self.set_gamma(gamma)
        self.load_params('model_parameters.npz')

    def calculate_variances(self, X, X_star):

        # YOUR CODE HERE
        gamma = self.gamma
        I = np.identity(self.m)
        phi = self.build_phi_matrix(X)
        phi_star = self.build_phi_matrix(X_star)
        Sigma_Y_star = phi_star @ (np.linalg.inv(phi.T @ phi + gamma * I)) @ phi_star.T
        return Sigma_Y_star

