"""Probabilistic Movement Primitive (ProMP)
========================================

ProMPs represent distributions of trajectories.
"""
import numpy as np


class ProMP:
    """Probabilistic Movement Primitive (ProMP).

    ProMPs have been proposed first in [1]_ and have been used later in [2]_,
    [3]_. The learning algorithm is a specialized form of the one presented in
    [4]_.

    Note that internally we represented trajectories with the task space
    dimension as the first axis and the time step as the second axis while
    the exposed trajectory interface is transposed. In addition, we internally
    only use a 1d array representation to make handling of the covariance
    simpler.

    Parameters
    ----------
    n_dims : int
        State space dimensions.

    n_weights_per_dim : int, optional (default: 10)
        Number of weights of the function approximator per dimension.

    References
    ----------
    .. [1] Paraschos, A., Daniel, C., Peters, J., Neumann, G. (2013).
       Probabilistic movement primitives, In C.J. Burges and L. Bottou and M.
       Welling and Z. Ghahramani and K.Q. Weinberger (Eds.), Advances in Neural
       Information Processing Systems, 26,
       https://papers.nips.cc/paper/2013/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf

    .. [3] Maeda, G. J., Neumann, G., Ewerton, M., Lioutikov, R., Kroemer, O.,
       Peters, J. (2017). Probabilistic movement primitives for coordination of
       multiple human–robot collaborative tasks. Autonomous Robots, 41, 593-612.
       DOI: 10.1007/s10514-016-9556-2,
       https://link.springer.com/article/10.1007/s10514-016-9556-2

    .. [2] Paraschos, A., Daniel, C., Peters, J., Neumann, G. (2018).
       Using probabilistic movement primitives in robotics. Autonomous Robots,
       42, 529-551. DOI: 10.1007/s10514-017-9648-7,
       https://www.ias.informatik.tu-darmstadt.de/uploads/Team/AlexandrosParaschos/promps_auro.pdf

    .. [4] Lazaric, A., Ghavamzadeh, M. (2010).
       Bayesian Multi-Task Reinforcement Learning. In Proceedings of the 27th
       International Conference on International Conference on Machine Learning
       (ICML'10) (pp. 599-606). https://hal.inria.fr/inria-00475214/document
    """
    def __init__(self, n_dims, n_weights_per_dim=10):
    """ 初始化promp参数

        参数：
            n_dims:状态空间维度（如机器人末端坐标的x/y/z）
            n_weights_per_dim:每个维度的基函数权重数量（默认10）“”“
      
        self.n_dims = n_dims    #状态维度
        self.n_weights_per_dim = n_weights_per_dim    #每个维度的权重数

        self.n_weights = n_dims * n_weights_per_dim    #总权重数（所有维度）
        
        #初始化权重的均值（0向量）和协方差（单位矩阵）
        self.weight_mean = np.zeros(self.n_weights)
        self.weight_cov = np.eye(self.n_weights)

        #RBF基函数的中心（在【0，1】区间均匀分布）
        self.centers = np.linspace(0, 1, self.n_weights_per_dim)

    def weights(self, T, Y, lmbda=1e-12):
        """Obtain ProMP weights by linear regression.
        从演示轨迹估计Promp权重（线性回归）
        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps     时间步

        Y : array-like, shape (n_steps, n_dims)
            Demonstrated trajectory     演示轨迹

        lmbda : float, optional (default: 1e-12)
            Regularization coefficient    正则化系数（防止过拟合）

        Returns
        -------
        weights : array, shape (n_steps * n_weights_per_dim)
            ProMP weights    估计的权重向量
        """
        #计算RBF的激活量（转置后用于回归）
        activations = self._rbfs_nd_sequence(T).T
        # 岭回归求解权重（(X^T X + λI)^-1 X^T Y）
        weights = np.linalg.pinv(
            activations.T.dot(activations)
            + lmbda * np.eye(activations.shape[1])
        ).dot(activations.T).dot(Y.T.ravel())    # Y展平为1D向量
        return weights

    def trajectory_from_weights(self, T, weights):
        """Generate trajectory from ProMP weights.
            从权重生成轨迹
        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        weights : array-like, shape (n_steps * n_weights_per_dim)
            ProMP weights

        Returns
        -------
        Y : array, shape (n_steps, n_dims)
            Trajectory    返回生成的轨迹
        """
        #权重与RBF激活值相乘，在重塑为轨迹形状
        return self._rbfs_nd_sequence(T).T.dot(weights).reshape(
            self.n_dims, len(T)).T    

    def condition_position(self, y_mean, y_cov=None, t=0, t_max=1.0):
        """Condition ProMP on a specific position.
            通过指定位置约束Promp（贝叶斯更新）
        For details, see page 4 of [1]_
            y_mean: 约束位置的均值（shape: [n_dims]）
            y_cov: 约束位置的协方差（默认0，表示硬约束）
            t: 约束对应的时间点
            t_max: 轨迹总时长（用于归一化时间）
        Parameters
        ----------
        y_mean : array, shape (n_dims,)
            Position mean.

        y_cov : array, shape (n_dims, n_dims), optional (default: 0)
            Covariance of position.

        t : float, optional (default: 0)
            Time at which the activations of RBFs will be queried. Note that
            we internally normalize the time so that t_max == 1.

        t_max : float, optional (default: 1)
            Duration of the ProMP

        Returns
        -------
        conditional_promp : ProMP
            New conditional ProMP    约束后新的promp实例

        References
        ----------
        .. [1] Paraschos, A., Daniel, C., Peters, J., Neumann, G. (2013).
           Probabilistic movement primitives, In C.J. Burges and L. Bottou and
           M. Welling and Z. Ghahramani and K.Q. Weinberger (Eds.), Advances in
           Neural Information Processing Systems, 26,
           https://papers.nips.cc/paper/2013/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
        """
        #计算时间t处的RBF激活值（块对角矩阵形式，适配多维度）
        Psi_t = _nd_block_diagonal(
            self._rbfs_1d_point(t, t_max)[:, np.newaxis], self.n_dims)
        if y_cov is None:
            y_cov = 0.0
        #贝叶斯更新公式（参考原论文的公式5和6）
        common_term = self.weight_cov.dot(Psi_t).dot(
            np.linalg.inv(y_cov + Psi_t.T.dot(self.weight_cov).dot(Psi_t)))

        # Equation (5)    更新权重均值
        weight_mean = (
            self.weight_mean
            + common_term.dot(y_mean - Psi_t.T.dot(self.weight_mean)))
        # Equation (6)    更新权重协方差
        weight_cov = (
            self.weight_cov - common_term.dot(Psi_t.T).dot(self.weight_cov))

        #创建新的promp实例存储约束后的分布
        conditional_promp = ProMP(self.n_dims, self.n_weights_per_dim)
        conditional_promp.from_weight_distribution(weight_mean, weight_cov)
        return conditional_promp

    def mean_trajectory(self, T):#计算均值的轨迹（从权重轨迹生成）
        """Get mean trajectory of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        Y : array, shape (n_steps, n_dims)
            Mean trajectory
        """
        return self.trajectory_from_weights(T, self.weight_mean)

    def cov_trajectory(self, T):#计算轨迹的协方差矩阵
        """Get trajectory covariance of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        cov : array, shape (n_dims * n_steps, n_dims * n_steps)
            Covariance
        """
        activations = self._rbfs_nd_sequence(T)
        return activations.T.dot(self.weight_cov).dot(activations)

    def var_trajectory(self, T):#计算轨迹各点的方差（协方差矩阵的对角线）
        """Get trajectory variance of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        var : array, shape (n_steps, n_dims)
            Variance
        """
        return np.maximum(np.diag(self.cov_trajectory(T)).reshape(
            self.n_dims, len(T)).T, 0.0)

    def mean_velocities(self, T):
        """Get mean velocities of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        Yd : array, shape (n_steps, n_dims)
            Mean velocities
        """
        return self._rbfs_derivative_nd_sequence(
            T).T.dot(self.weight_mean).reshape(self.n_dims, len(T)).T

    def cov_velocities(self, T):
        """Get velocity covariance of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        cov : array, shape (n_dims * n_steps, n_dims * n_steps)
            Covariance
        """
        activations = self._rbfs_derivative_nd_sequence(T)
        return activations.T.dot(self.weight_cov).dot(activations)

    def var_velocities(self, T):
        """Get velocity variance of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        var : array, shape (n_steps, n_dims)
            Variance
        """
        return np.maximum(np.diag(self.cov_velocities(T)).reshape(
            self.n_dims, len(T)).T, 0.0)

    def sample_trajectories(self, T, n_samples, random_state):
        """Sample trajectories from ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        n_samples : int
            Number of trajectories that will be sampled

        random_state : np.random.RandomState
            State of random number generator

        Returns
        -------
        samples : array, shape (n_samples, n_steps, n_dims)
            Sampled trajectories
        """
        weight_samples = random_state.multivariate_normal(
            self.weight_mean, self.weight_cov, n_samples)
        samples = np.empty((n_samples, len(T), self.n_dims))
        for i in range(n_samples):
            samples[i] = self.trajectory_from_weights(T, weight_samples[i])
        return samples

    def from_weight_distribution(self, mean, cov):
        """Initialize ProMP from mean and covariance in weight space.

        Parameters
        ----------
        mean : array, shape (n_dims * n_weights_per_dim)
            Mean of weight distribution

        cov : array, shape (n_dims * n_weights_per_dim, n_dims * n_weights_per_dim)
            Covariance of weight distribution

        Returns
        -------
        self : ProMP
            This object
        """
        self.weight_mean = mean
        self.weight_cov = cov
        return self

    def imitate(self, Ts, Ys, n_iter=1000, min_delta=1e-5, verbose=0):
        r"""Learn ProMP from multiple demonstrations.
            从多个演示轨迹学习promp（EM算法）
        For details, see Section 3.2 of [1]_. We use the parameters
        :math:`P = I` (identity matrix), :math:`\mu_0 = 0, k_0 = 0, \nu_0 = 0,
        \Sigma_0 = 0,\alpha_0 = 0,\beta_0 = 0`.

        Parameters
        ----------
         Ts: 演示轨迹的时间步（shape: [n_demos, n_steps]）
         Ys: 演示轨迹数据（shape: [n_demos, n_steps, n_dims]）
         n_iter: 最大迭代次数
         min_delta: 收敛阈值（均值变化小于该值时停止）
         
        Ts : array, shape (n_demos, n_steps)
            Time steps of demonstrations

        Ys : array, shape (n_demos, n_steps, n_dims)
            Demonstrations

        n_iter : int, optional (default: 1000)
            Maximum number of iterations

        min_delta : float, optional (default: 1e-5)
            Minimum delta between means to continue iteration

        verbose : int, optional (default: 0)
            Verbosity level

        References
        ----------
        .. [1] Lazaric, A., Ghavamzadeh, M. (2010).
           Bayesian Multi-Task Reinforcement Learning. In Proceedings of the
           27th International Conference on International Conference on Machine
           Learning (ICML'10) (pp. 599-606).
           https://hal.inria.fr/inria-00475214/document
        """
        #初始化参数（gamma为平滑系数，variance为噪声方差）
        gamma = 0.7

        n_demos = len(Ts)
        self.variance = 1.0

        #存储每个轨迹的权重均值和协方差
        means = np.zeros((n_demos, self.n_weights))
        covs = np.empty((n_demos, self.n_weights, self.n_weights))

        # Precompute constant terms in expectation-maximization algorithm
        #预计算EM算法中的常量（如RBF激活值、轨迹平滑矩阵）
        # n_demos x n_steps*self.n_dims x n_steps*self.n_dims
        Hs = []
        for demo_idx in range(n_demos):
            n_steps = len(Ys[demo_idx])
            H_partial = np.eye(n_steps)
            for y in range(n_steps - 1):
                H_partial[y, y + 1] = -gamma
            H = _nd_block_diagonal(H_partial, self.n_dims)
            Hs.append(H)

        # n_demos x n_steps*n_dims
        Ys_rearranged = [Y.T.ravel() for Y in Ys]

        # n_demos x n_steps*self.n_dims
        Rs = []
        for demo_idx in range(n_demos):
            R = Hs[demo_idx].dot(Ys_rearranged[demo_idx])
            Rs.append(R)

        # n_demos
        # RR in original code
        RTRs = []
        for demo_idx in range(n_demos):
            RTR = Rs[demo_idx].T.dot(Rs[demo_idx])
            RTRs.append(RTR)

        # n_demos x self.n_dims*self.n_weights_per_dim
        # x self.n_dims*self.n_steps
        # BH in original code
        PhiHTs = []
        for demo_idx in range(n_demos):
            PhiHT = self._rbfs_nd_sequence(Ts[demo_idx]).dot(Hs[demo_idx].T)
            PhiHTs.append(PhiHT)

        # n_demos x self.n_dims*self.n_weights_per_dim
        # mean_esteps in original code
        PhiHTRs = []
        for demo_idx in range(n_demos):
            PhiHTR = PhiHTs[demo_idx].dot(Rs[demo_idx])
            PhiHTRs.append(PhiHTR)

        # n_demos x self.n_dims*self.n_weights_per_dim
        # x self.n_dims*self.n_weights_per_dim
        # cov_esteps in original code
        PhiHTHPhiTs = []
        for demo_idx in range(n_demos):
            PhiHTHPhiT = PhiHTs[demo_idx].dot(PhiHTs[demo_idx].T)
            PhiHTHPhiTs.append(PhiHTHPhiT)

        n_samples = sum([Y.shape[0] for Y in Ys])
        
        #EM迭代
        for it in range(n_iter):
            weight_mean_old = self.weight_mean
            #E步：估计每个演示的权重分布
            for demo_idx in range(n_demos):
                means[demo_idx], covs[demo_idx] = self._expectation(
                        PhiHTRs[demo_idx], PhiHTHPhiTs[demo_idx])
            #M步：更新promp的权重先验分布和噪声方差
            self._maximization(
                means, covs, RTRs, PhiHTRs, PhiHTHPhiTs, n_samples)
            #检查收敛
            delta = np.linalg.norm(self.weight_mean - weight_mean_old)
            if verbose:
                print("Iteration %04d: delta = %g" % (it + 1, delta))
            if delta < min_delta:
                break

    def _rbfs_1d_point(self, t, t_max=1.0, overlap=0.7):
        """Radial basis functions for one dimension and a point.
            计算单个时间点的1D RBF激活值
        Parameters
        ----------
        t : float
            Time at which the activations of RBFs will be queried. Note that
            we internally normalize the time so that t_max == 1.

        t_max : float, optional (default: 1)
            Duration of the ProMP

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        Returns
        -------
        activations : array, shape (n_weights_per_dim,)
            Activations of RBFs for each time step.
        """
        #计算RBF带宽（根据重叠率overlap确定）
        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        # normalize time to interval [0, 1]#归一化时间到【0，1】
        t = t / t_max

        activations = np.exp(-(t - self.centers[:]) ** 2 / (2.0 * h))
        activations /= activations.sum(axis=0)  # normalize activations for each step     归一化和为1

        assert activations.ndim == 1
        assert activations.shape[0] == self.n_weights_per_dim

        return activations

    def _rbfs_nd_sequence(self, T, overlap=0.7):
        """Radial basis functions for n_dims dimensions and a sequence.
        
        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Times at which the activations of RBFs will be queried. Note that
            we assume that T[0] == 0.0 and the times will be normalized
            internally so that T[-1] == 1.0.

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        Returns
        -------
        activations : array, shape (n_dims * n_weights_per_dim, n_dims * n_steps)
            Activations of RBFs for each time step and each dimension.
        """
        return _nd_block_diagonal(
            self._rbfs_1d_sequence(T, overlap), self.n_dims)

    def _rbfs_1d_sequence(self, T, overlap=0.7, normalize=True):
        """Radial basis functions for one dimension and a sequence.
            计算时间序列的1D RBF激活值(shape: [n_weights_per_dim,n_steps])
        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Times at which the activations of RBFs will be queried. Note that
            we assume that T[0] == 0.0 and the times will be normalized
            internally so that T[-1] == 1.0.

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        normalize : bool, optional (default: True)
            Normalize activations to sum up to one in each step

        Returns
        -------
        activations : array, shape (n_weights_per_dim, n_steps)
            Activations of RBFs for each time step.
        """
        assert T.ndim == 1

        n_steps = len(T)

        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        # normalize time to interval [0, 1]
        T = np.atleast_2d(T)
        T /= np.max(T)

        activations = np.exp(
            -(T - self.centers[:, np.newaxis]) ** 2 / (2.0 * h))
        if normalize:
            activations /= activations.sum(axis=0)

        assert activations.shape[0] == self.n_weights_per_dim
        assert activations.shape[1] == n_steps

        return activations

    def _rbfs_derivative_nd_sequence(self, T, overlap=0.7):
        """Derivative of RBFs for n_dims dimensions and a sequence.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Times at which the activations of RBFs will be queried. Note that
            we assume that T[0] == 0.0 and the times will be normalized
            internally so that T[-1] == 1.0.

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        Returns
        -------
        activations : array, shape (n_dims * n_weights_per_dim, n_dims * n_steps)
            Activations of derivative of RBFs for each time step and dimension.
        """
        return _nd_block_diagonal(
            self._rbfs_derivative_1d_sequence(T, overlap), self.n_dims)

    def _rbfs_derivative_1d_sequence(self, T, overlap=0.7):
        """Derivative of RBFs for one dimension and a sequence.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Times at which the activations of RBFs will be queried. Note that
            we assume that T[0] == 0.0 and the times will be normalized
            internally so that T[-1] == 1.0.

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        Returns
        -------
        activations : array, shape (n_weights_per_dim, n_steps)
            Activations of derivative of RBFs for each time step.
        """
        assert T.ndim == 1

        n_steps = len(T)

        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        rbfs = self._rbfs_1d_sequence(T, overlap, normalize=False)
        rbfs_sum_per_step = rbfs.sum(axis=0)

        # normalize time to interval [0, 1]
        T = np.atleast_2d(T)
        T /= np.max(T)

        rbfs_deriv = (self.centers[:, np.newaxis] - T) / h
        rbfs_deriv *= rbfs
        rbfs_deriv_sum_per_step = rbfs_deriv.sum(axis=0)
        rbfs_deriv = (
             rbfs_deriv * rbfs_sum_per_step
             - rbfs * rbfs_deriv_sum_per_step) / (rbfs_sum_per_step ** 2)

        assert rbfs_deriv.shape[0] == self.n_weights_per_dim
        assert rbfs_deriv.shape[1] == n_steps

        return rbfs_deriv

    def _expectation(self, PhiHTR, PhiHTHPhiT):
        cov = np.linalg.pinv(PhiHTHPhiT / self.variance
                             + np.linalg.pinv(self.weight_cov))
        mean = cov.dot(PhiHTR / self.variance
                       + np.linalg.pinv(self.weight_cov).dot(self.weight_mean))
        return mean, cov

    def _maximization(self, means, covs, RRs, PhiHTR, PhiHTHPhiTs, n_samples):
        M = len(means)

        self.weight_mean = np.mean(means, axis=0)

        centered = means - self.weight_mean
        self.weight_cov = (centered.T.dot(centered) + np.sum(covs, axis=0)) / M
        # TODO what is d + 2?

        self.variance = 0.0
        for i in range(len(means)):
            # trace is the same, independent of the order of matrix
            # multiplications, see:
            # https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf,
            # Equation 16
            self.variance += np.trace(PhiHTHPhiTs[i].dot(covs[i]))

            self.variance += RRs[i]
            self.variance -= 2.0 * PhiHTR[i].T.dot(means[i].T)
            self.variance += (means[i].dot(PhiHTHPhiTs[i].dot(means[i].T)))

        # TODO why these factors?
        self.variance /= (np.linalg.norm(means) * M * self.n_dims * n_samples
                          + 2.0)
        #self.variance /= self.n_dims * n_samples + 2.0


def _nd_block_diagonal(partial_1d, n_dims):
    """Replicates matrix n_dims times to form a block-diagonal matrix.
        生成块对角矩阵（将1D RBF激活值扩展到多维度）
    We also accept matrices of rectangular shape. In this case the result is
    not officially called a block-diagonal matrix anymore.

    Parameters
    ----------
    partial_1d : array, shape (n_block_rows, n_block_cols)
        Matrix that should be replicated.

    n_dims : int
        Number of times that the matrix has to be replicated.

    Returns
    -------
    full_nd : array, shape (n_block_rows * n_dims, n_block_cols * n_dims)
        Block-diagonal matrix with n_dims replications of the initial matrix.
    """
    assert partial_1d.ndim == 2
    n_block_rows, n_block_cols = partial_1d.shape

    full_nd = np.zeros((n_block_rows * n_dims, n_block_cols * n_dims))
    for j in range(n_dims):
        full_nd[n_block_rows * j:n_block_rows * (j + 1),
                n_block_cols * j:n_block_cols * (j + 1)] = partial_1d
    return full_nd


def via_points(promp, ts, y_cond, y_conditional_cov=None):
    """Condition ProMP on several via-points.
    
        通过多个中间点约束promp（多次调用condition_position)
        
    For details, see section 2.2 on page 4 of [1]_

    Parameters
    ----------
    promp : ProMP instance

    ts : array, shape (n_positions,)
        Time vector at which the activations of RBFs will be queried. Note that
        we internally normalize the time so that t_max == 1.

    y_cond : array, shape (n_positions,)
        Desired mean position vector corresponding to each time in `ts`.

    y_conditional_cov : array, shape (n_dims, n_dims), optional (default: 0)
        Covariances of desired positions.

    Returns
    -------
    conditional_promp : ProMP
        New conditional ProMP

    References
    ----------
    .. [1] Paraschos, A., Daniel, C., Peters, J., Neumann, G. (2013).
       Probabilistic movement primitives, In C.J. Burges and L. Bottou and
       M. Welling and Z. Ghahramani and K.Q. Weinberger (Eds.), Advances in
       Neural Information Processing Systems, 26,
       https://papers.nips.cc/paper/2013/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
    """
    if y_conditional_cov is None:
        y_conditional_cov = np.zeros(y_cond.shape)

    for idx, t_i in enumerate(ts):
        promp = promp.condition_position(
            y_mean=y_cond[idx],
            y_cov=y_conditional_cov[[idx]],
            t=t_i,
            t_max=1.0
        )
    return promp
