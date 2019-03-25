import numpy as np
import cvxpy as cvx
from matplotlib import pyplot as plt


def gaussian_kernel(x, y):
    """
    :param x: [x samples x n dimensions]
    :param y: [y samples x n dimensions]
    :return: K(x,y) [y samples x x samples]
    """

    # initialize kernel matrix
    k = np.zeros([y.shape[0], x.shape[0]])

    # loop over rows
    for i in range(k.shape[0]):

        k[i] = np.exp(-0.5 * np.linalg.norm(x - np.expand_dims(y[i], axis=0), ord=2, axis=1) ** 2)

    return k


class LogisticRegression:

    def __init__(self):

        # model name
        self.name = 'LR'
        self.linestyle = ':'
        self.color = 'g'

        # prediction decision threshold
        self.pred_thresh = 0.5

        # model parameters
        self.w = None
        self.b = None

    def hyper_parameters(self):

        # generate hyper-parameter test space
        gamma = np.logspace(0, 3, 4)

        # return list of parameter dictionaries
        p = []
        for i in range(len(gamma)):
            p.append({'gamma': gamma[i]})

        return p

    def train(self, x, y, gamma=0.1):

        # ensure labels are [0, 1]
        y[y == -1] = 0
        assert np.unique(y).tolist() == [0, 1]

        # regularization parameter
        g = cvx.Parameter(sign="positive")
        g.value = gamma / x.shape[1]

        # define model variables
        w = cvx.Variable(x.shape[1])
        b = cvx.Variable()

        # compute affine transform
        a = x * w - b

        # compute log-likelihood
        l = cvx.sum_entries(cvx.mul_elemwise(y, a)) - cvx.sum_entries(cvx.logistic(a))

        # minimize negative log-likelihood plus l-2 regularization
        obj = cvx.Minimize(-l + g * cvx.sum_squares(w))

        try:

            # form problem and solve
            prob = cvx.Problem(obj)
            prob.solve()

            # throw error if not optimal
            assert prob.status == 'optimal'

            # save model parameters
            self.w = np.array(w.value)
            self.b = np.array(b.value)

            # return success
            return True

        except:

            # return failure
            return False

    def predict(self, x):

        # make prediction
        y_hat = np.squeeze(np.sign(x @ self.w - self.b - self.pred_thresh))

        return y_hat


class SVM:

    def __init__(self):

        # model name
        self.name = 'svm'
        self.linestyle = '--'
        self.color = 'b'

        # prediction decision threshold
        self.pred_thresh = 0

        # model parameters
        self.w = None
        self.b = None

    def hyper_parameters(self):

        # generate hyper-parameter test space
        C = np.logspace(-4, 1, 6)

        # return list of parameter dictionaries
        p = []
        for i in range(len(C)):
            p.append({'C': C[i]})

        return p

    def train(self, x, y, C=1, mode='primal'):

        # ensure labels are [-1, 1]
        y[y == 0] = -1
        assert np.unique(y).tolist() == [-1, 1]

        # ensure mode is one of expected values
        assert mode == 'primal' or mode == 'dual' or mode == 'test'

        # regularization parameter
        c = cvx.Parameter(sign="positive")
        c.value = C

        # running primal or in test mode
        if mode == 'primal' or mode == 'test':

            # define primal model variables
            w = cvx.Variable(x.shape[1])
            b = cvx.Variable()
            u = cvx.Variable(x.shape[0])

            # define primal objective
            obj = cvx.Minimize(0.5 * cvx.sum_squares(w) + c * cvx.sum_entries(u))

            # define primal constraints
            constraints = [cvx.mul_elemwise(y, (x * w - b)) >= 1 - u,
                           u >= 0]

            try:

                # form problem and solve
                prob = cvx.Problem(obj, constraints)
                prob.solve()

                # throw error if not optimal
                assert prob.status == 'optimal'

                # save model parameters
                self.w = np.array(w.value)
                self.b = np.array(b.value)

                # return success
                return True

            except:

                # return failure
                return False

        # running dual problem or test
        if mode == 'dual' or mode == 'test':

            # define dual model variables
            alpha = cvx.Variable(x.shape[0])

            # define dual objective
            xy = x * np.expand_dims(y, axis=-1)
            P = xy @ xy.T
            obj = cvx.Maximize(cvx.sum_entries(alpha) - 0.5 * cvx.quad_form(alpha, P))

            # define dual constraints
            constraints = [cvx.sum_entries(cvx.mul_elemwise(y, alpha)) == 0,
                           alpha >= 0,
                           c >= alpha]

            # form problem and solve
            prob = cvx.Problem(obj, constraints)
            prob.solve()

            # throw error if not optimal
            assert prob.status == 'optimal' or prob.status == 'optimal_inaccurate'

            # grab alpha as array
            alpha = np.squeeze(np.array(alpha.value))

            # convert back to primal variables
            w = np.expand_dims(np.sum(np.expand_dims(alpha * y, axis=-1) * x, axis=0), axis=-1)
            b = -np.sum((y - np.squeeze(x @ w)) * alpha) / np.sum(alpha)

            # test mode
            if mode == 'test':

                assert (np.abs(self.w - w) < 1e-3).all()
                # assert np.abs(self.b - b) < 1

            # regular dual
            else:

                # save model variables
                self.w = w
                self.b = b

        return prob.value

    def predict(self, x):

        # make prediction
        y_hat = np.squeeze(np.sign(x @ self.w - self.b - self.pred_thresh))

        return y_hat


class SVMPlus:

    def __init__(self):

        # model name
        self.name = 'svm+'
        self.linestyle = '-'
        self.color = 'r'

        # prediction decision threshold
        self.pred_thresh = 0

        # model parameters
        self.w = None
        self.b = None

    def hyper_parameters(self):

        # generate hyper-parameter test space
        C = np.logspace(-6, -5, 4)
        gamma = np.logspace(0, 3, 4)

        # return list of parameter dictionaries
        p = []
        for i in range(len(C)):
            for j in range(len(gamma)):
                p.append({'C': C[i], 'gamma': gamma[j]})

        return p

    def train(self, x, y, x_star=None, gamma=1.0, C=1.0):

        # ensure labels are [-1, 1]
        y[y == 0] = -1
        assert np.unique(y).tolist() == [-1, 1]

        # if x* not supplied just make it ones
        if x_star is None:
            x_star = np.ones([x.shape[0], 1])

        # regularization parameter
        g = cvx.Parameter(sign="positive")
        g.value = gamma
        c = cvx.Parameter(sign="positive")
        c.value = C

        # define model variables
        w = cvx.Variable(x.shape[1])
        b = cvx.Variable()
        w_star = cvx.Variable(x_star.shape[1])
        d = cvx.Variable()

        # define objective
        obj = cvx.Minimize(0.5 * g * cvx.sum_squares(w) / x.shape[1] +
                           0.5 * g * cvx.sum_squares(w_star) / x_star.shape[1] +
                           c * cvx.sum_entries(x_star * w_star - d) / x.shape[0])

        # define constraints
        constraints = [cvx.mul_elemwise(y, (x * w - b)) >= 1 - (x_star * w_star - d),
                       (x_star * w_star - d) >= 0]

        # form problem and solve
        prob = cvx.Problem(obj, constraints)
        prob.solve()

        try:

            # throw error if not optimal
            assert prob.status == 'optimal'

            # save model parameters
            self.w = np.array(w.value)
            self.b = np.array(b.value)

            # return success
            return True

        except:

            # return failure
            return False

    def predict(self, x):

        # make prediction
        y_hat = np.squeeze(np.sign(x @ self.w - self.b - self.pred_thresh))

        return y_hat


def simulate_linear(mdls, N=100):

    # define line
    def line(x):
        return -2 * x + 3

    # simulate 2 dimensions for first class
    x1_y0 = np.random.uniform(size=[N, 1])
    x2_y0 = line(x1_y0) + np.random.uniform(low=-0.01, high=3.0, size=[N, 1])
    x_y0 = np.concatenate([x1_y0, x2_y0], axis=1)

    # simulate 2 dimensions for second class
    x1_y1 = np.random.uniform(size=[N, 1])
    x2_y1 = line(x1_y1) - np.random.uniform(low=-0.01, high=3.0, size=[N, 1])
    x_y1 = np.concatenate([x1_y1, x2_y1], axis=1)

    # aggregate the data
    x = np.concatenate([x_y0, x_y1])
    y = np.concatenate([-np.ones(N), np.ones(N)])

    # get the max and min
    x1_min = np.min(x[:, 0])
    x1_max = np.max(x[:, 0])

    # plot the data
    plt.figure()
    plt.scatter(x[y == -1, 0], x[y == -1, 1], label='y=0')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], label='y=1')
    plt.legend()

    # loop over the models
    for mdl in mdls:

        # train the model
        mdl.train(x, y)

        # plot resulting decision boundary: 0 = x * w - b = x0 * w0 + x1 * w1 - b = 0
        x2_min = (mdl.pred_thresh + mdl.b - mdl.w[0, 0] * x1_min) / mdl.w[1, 0]
        x2_max = (mdl.pred_thresh + mdl.b - mdl.w[0, 0] * x1_max) / mdl.w[1, 0]
        plt.plot([x1_min, x1_max], [x2_min, x2_max], color='k', linestyle=mdl.linestyle, label=mdl.name)

    # plot the legend
    plt.legend()


def simulate_gaussian(mdls, N=100):

    # define polar to cartesian coordinate
    def polar_to_cartesian(r, theta):
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        return np.concatenate([x1, x2], axis=1)

    # sample an angle and a radius for first class
    r_y0 = np.random.uniform(0, 1, size=[N, 1])
    theta_y0 = np.random.uniform(0, 2 * np.pi, size=[N, 1])
    x_y0 = polar_to_cartesian(r_y0, theta_y0)

    # sample an angle and a radius for first class
    r_y1 = np.random.uniform(0.75, 1.5, size=[N, 1])
    theta_y1 = np.random.uniform(0, 2 * np.pi, size=[N, 1])
    x_y1 = polar_to_cartesian(r_y1, theta_y1)

    # aggregate the data
    x = np.concatenate([x_y0, x_y1])
    y = np.concatenate([-np.ones(N), np.ones(N)])

    # plot the data
    plt.figure()
    plt.scatter(x[y == -1, 0], x[y == -1, 1], label='y=0')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], label='y=1')

    # loop over the models
    for mdl in mdls:

        # generate kernel matrix
        k = gaussian_kernel(x, x)

        # train the model in kernel space
        mdl.train(k, y)

        # plot resulting by generating mesh grid
        x1 = np.arange(np.min(x[:, 0]), np.max(x[:, 0]), 0.05)
        x2 = np.arange(np.min(x[:, 1]), np.max(x[:, 1]), 0.05)
        x1_g, x2_g = np.meshgrid(x1, x2)
        y_pred = np.zeros([x1_g.shape[0], x1_g.shape[1]])
        for i in range(x1_g.shape[0]):
            for j in range(x1_g.shape[1]):

                # generate kernel
                x_g = np.array([x1_g[i, j], x2_g[i, j]]).reshape([1, 2])
                k = gaussian_kernel(x, x_g)

                # make prediction
                y_pred[i, j] = mdl.predict(k)

        # plot decision contours
        c = plt.contour(x1_g, x2_g, y_pred, levels=[0])
        c.collections[0].set_label(mdl.name)
        c.collections[0].set_color(mdl.color)
        # c.collections[0].set_linestyle(mdl.linestyle)
        c.collections[0].set_alpha(0.8)

    # show the legend
    plt.legend()


if __name__ == '__main__':

    # print installed solvers
    print(cvx.installed_solvers())

    # simulate linear separable data
    simulate_linear([LogisticRegression(), SVM(), SVMPlus()])

    # test gaussian kernel for all three models
    simulate_gaussian([LogisticRegression(), SVM(), SVMPlus()])

    # show the plots
    plt.show()
