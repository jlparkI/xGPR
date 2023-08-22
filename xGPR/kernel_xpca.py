"""This tool enables easy generation of a kernel PCA plot using
any of the available xGPR kernels. It makes use of the preconditioner
construction routines to generate the kPCA. If you have trained an
xGPR model, you can use the hyperparameters that you used for
the xGPR model for the kPCA (without this, you will have to rely
on heuristics to choose good hyperparameters, similar to UMAP
and T-SNE)."""
import numpy as np
import scipy
try:
    import cupy as cp
    import cupyx
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")

from .constants import constants
from .auxiliary_baseclass import AuxiliaryBaseclass



class KernelxPCA(AuxiliaryBaseclass):
    """A tool for generating kernel PCA plots
    using random features."""

    def __init__(self, num_rffs, hyperparams, dataset, n_components = 2,
            kernel_choice = "RBF", device = "cpu",
            kernel_specific_params = constants.DEFAULT_KERNEL_SPEC_PARMS,
            random_seed = 123, verbose = True, num_threads = 2,
            double_precision_fht = False):
        """The constructor.

        Args:
            num_rffs (int): The number of random Fourier features
                to use for the auxiliary device.
            dataset: A valid dataset object.
            hyperparams (np.ndarray): A numpy array containing the kernel-specific
                hyperparameter. If you have fitted an xGPR model, the first two
                hyperparameters are general not kernel specific, so
                my_model.get_hyperparams()[2:] will retrieve the hyperparameters you
                need. For most kernels there is only one kernel-specific hyperparameter.
                For kernels with no kernel-specific hyperparameter (e.g. arc-cosine
                and polynomial kernels), this argument is ignored.
            n_components (int): The number of PCA components to generate. For visualization,
                2 is the logical choice. There are some situations where more components
                might make sense (if using them as input to some other algorithm).
            kernel_choice (str): The kernel that the model will use.
                Must be in kernels.kernel_list.KERNEL_NAME_TO_CLASS.
            device (str): Determines whether calculations are performed on
                'cpu' or 'gpu'. The initial entry can be changed later
                (i.e. model can be transferred to a different device).
                Defaults to 'cpu'.
            kernel_specific_params (dict): Contains kernel-specific parameters --
                e.g. 'matern_nu' for the nu for the Matern kernel, or 'conv_width'
                for the conv1d kernel.
            random_seed (int): A seed for the random number generator.
            verbose (bool): If True, regular updates are printed
                during fitting and tuning. Defaults to True.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. If running on GPU, this argument is ignored.
            double_precision_fht (bool): If True, use double precision during FHT for
                generating random features. For most problems, it is not beneficial
                to set this to True -- it merely increases computational expense
                with negligible benefit -- but this option is useful for testing.
                Defaults to False.
        """
        super().__init__(num_rffs, hyperparams, dataset,
                        kernel_choice, device, kernel_specific_params,
                        random_seed, verbose, num_threads,
                        double_precision_fht)
        self.n_components = n_components

        #Initialize the kPCA model.
        dataset.device = self.device
        self.z_mean = self.get_mapped_data_statistics(dataset)
        self.eigvecs = self.initialize_kpca(dataset)
        dataset.device = "cpu"


    def predict(self, input_x, chunk_size = 2000):
        """Generates a low-dimensional projection of the input."""
        xdata = self.pre_prediction_checks(input_x)
        preds = []
        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            xfeatures = self.kernel.transform_x(xdata[i:cutoff, :])
            xfeatures -= self.z_mean[None,:]
            preds.append(xfeatures @ self.eigvecs)

        if self.device == "gpu":
            preds = cp.asnumpy(cp.vstack(preds))
        else:
            preds = np.vstack(preds)
        return preds


    def get_mapped_data_statistics(self, dataset):
        """This function gets the mean of the random feature-mapped
        data from the training dataset.

        Args:
            dataset: A dataset object that can return chunks of
                data.

        Returns:
            train_mean (array): An array storing the mean of the
                random feature mapped-training data.
        """
        ndpoints = 0
        if self.device == "gpu":
            train_mean = cp.zeros((self.kernel.get_num_rffs()))
        else:
            train_mean = np.zeros((self.kernel.get_num_rffs()))

        for x_data in dataset.get_chunked_x_data():
            if dataset.pretransformed:
                x_features = x_data
            else:
                x_features = self.kernel.transform_x(x_data)
            w_1 = x_data.shape[0] / (x_data.shape[0] + ndpoints)
            w_2 = ndpoints / (ndpoints + x_data.shape[0])
            x_features_mean = x_features.mean(axis=0)

            train_mean = w_1 * x_features_mean + w_2 * train_mean
            ndpoints += x_data.shape[0]

        return train_mean


    def single_pass_gauss(self, dataset, kernel, q_mat, acc_results):
        """Runs a single pass over the dataset using matvecs.
        Modifications are in-place.

        Args:
            dataset: A valid dataset object.
            q_mat (array): A (num_rffs, rank) array against
                which the random features will be multiplied.
            acc_results (array): A (num_rffs, rank) array
                in which Z^T Z @ q_mat will be stored.
        """
        if dataset.pretransformed:
            for xdata in dataset.get_chunked_x_data():
                x_trans = xdata - self.z_mean[None,:]
                acc_results += x_trans.T @ (x_trans @ q_mat)
        else:
            for xdata in dataset.get_chunked_x_data():
                xdata = kernel.transform_x(xdata)
                xdata -= self.z_mean[None,:]
                acc_results += xdata.T @ (xdata @ q_mat)


    def initialize_kpca(self, dataset):
        """Uses the Nystrom approximation of Z^T Z to find its
        (approximate) top two eigenvectors & eigenvalues.

        Args:
            dataset: An OnlineDataset or OfflineDataset containing the raw data.

        Returns:
            u_mat (np.ndarray): The eigenvectors of the matrix needed to
                form the preconditioner.
        """
        fitting_rffs = self.kernel.get_num_rffs()
        rng = np.random.default_rng(self.random_seed)
        l_mat = rng.standard_normal(size=(fitting_rffs, self.n_components))
        l_mat, _ = np.linalg.qr(l_mat)

        if self.kernel.device == "cpu":
            acc_results = np.zeros((self.kernel.get_num_rffs(), self.n_components))
            svd_calculator, cho_calculator = np.linalg.svd, np.linalg.cholesky
            tri_solver = scipy.linalg.solve_triangular
            qr_calculator = np.linalg.qr
        else:
            mempool = cp.get_default_memory_pool()
            acc_results = cp.zeros((self.kernel.get_num_rffs(), self.n_components))
            svd_calculator, cho_calculator = cp.linalg.svd, cp.linalg.cholesky
            tri_solver = cupyx.scipy.linalg.solve_triangular
            qr_calculator = cp.linalg.qr
            l_mat = cp.asarray(l_mat)

        self.single_pass_gauss(dataset, self.kernel, l_mat, acc_results)


        if self.device == "gpu":
            mempool.free_all_blocks()

        for i in range(2):
            q_mat, r_mat = qr_calculator(acc_results)
            acc_results[:] = 0.0
            del r_mat
            if self.device == "gpu":
                mempool.free_all_blocks()
            self.single_pass_gauss(dataset, self.kernel, q_mat, acc_results)

        if self.device == "gpu":
            mempool.free_all_blocks()

        norm = float( np.sqrt((acc_results**2).sum())  )

        shift = np.spacing(norm)
        acc_results += shift * q_mat
        q_mat = q_mat.T @ acc_results

        q_mat = cho_calculator(q_mat)

        if self.device == "gpu":
            mempool.free_all_blocks()

        acc_results = tri_solver(q_mat, acc_results.T,
                            overwrite_b = True, lower=True).T

        u_mat, _, _ = svd_calculator(acc_results, full_matrices=False)
        return u_mat


    @property
    def n_components(self):
        """Property definition."""
        return self._num_components

    @n_components.setter
    def n_components(self, value):
        """Setter for num_components."""
        if not isinstance(value, int):
            raise ValueError("Tried to set num_components using something that "
                    "was not an int!")
        if value < 1 or value >= self.num_rffs:
            raise ValueError("num_components must be > 0 and < num_rffs.")
        self._num_components = value
