class logistic_regression:

    def solve(self,x,y):
        # add column of ones to x for bias
        x = np.insert(x, 0, 1, axis=1)
        x_0 = x[y == -1]
        x_1 = x[y==1]
        # save problem dimensions
        m = x.shape[0]
        n = x.shape[1]
        w = cvx.Variable(n)
        expr1 = cvx.sum_entries(x_1*w)
        expr2 = -cvx.sum_entries(cvx.logistic(x*w))
        obj = cvx.Maximize(expr1+expr2)
        problem = cvx.Problem(obj)
        problem.solve()


        return problem.value,w.value

class gaussian_svm_classifier:

    def __init__(self, x,gamma=0.1):

        # save problem dimensions
        self.x = x
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.width = 2*np.std(x)
        # regularization parameter
        self.gamma = cvx.Parameter(sign="positive")
        self.gamma.value = gamma
    def kernel_fn(self,xi,xj):
        return np.exp((-1 / self.width) * ((np.linalg.norm(x_i - x_j)) ** 2))

    def problem(self, y):
        # ensure labels are [-1, 1]
        y[y == 0] = -1

        # define model variables
        self.alpha = cvx.Variable(self.m)
        # make gaussian kernel matrix
        #setting width parameter
        self.width = 2*np.std(x)
        K = np.zeros([x.shape[0], x.shape[0]])

        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[0]):
                x_i = x[i]
                x_j = x[j]
                exp_arg = (-1 / self.width) * ((np.linalg.norm(x_i - x_j)) ** 2)
                K[i, j] = np.exp(exp_arg)
        # define objective

        #(for gaussian SVM, we need to optimize the dual, because phi(x) is infinite dimensional)
        cvx.sum_entries(self.alpha)
        alpha_matrix = self.alpha*self.alpha.T
        y_matrix = y*y.T

        obj = cvx.Maximize(cvx.sum_entries(self.alpha)-(1/2)*cvx.sum_entries(alpha_matrix*y_matrix*K))

        # define constraints
        constraints = [cvx.sum_entries(cvx.multiply(self.alpha,y))==0,self.alpha<=self.gamma.value,self.alpha>=0]

        # form problem and solve
        prob = cvx.Problem(obj, constraints)
        prob.solve()

        # throw error if not optimal
        assert prob.status == 'optimal'

        # don't return anything

    def pred(self,x0):
        sum = 0
        for i in range(self.m):
            exp_arg = (-1 / self.width) * ((np.linalg.norm(x0 - self.x[i])) ** 2)
            kernel_val = np.exp(exp_arg)
            sum += (self.alpha.value[i])*kernel_val*self.y[i])
        #solving for bias b
        sum2 = 0
        for i in range(self.m):
            exp_arg = (-1 / self.width) * ((np.linalg.norm(self.x[0] - self.x[i])) ** 2)
            kernel_val = np.exp(exp_arg)
            sum2 += (self.alpha.value[i]) * kernel_val * self.y[i])
        b = self.y[0] - sum

        y0 = np.sign(sum + b)
        dist = abs(sum+b)

        return y0, dist


    def draw_dec_boundary(x):
        max_x0 = np.max(x[:,0])
        min_x0 = np.min(x[:,0])
        max_x1 = np.max(x[:,1])
        min_x1 = np.min(x[:,1])
        h = .02  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = self.pred(np.c_[xx.ravel(), yy.ravel()])




