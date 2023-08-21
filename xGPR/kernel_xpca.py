"""This experimental tool enables easy generation of a kernel PCA plot
using a fitted xGPR model. It makes use of the preconditioner
construction routines to generate the kPCA and is built and
used by 'plugging in' the xGPR model."""
import numpy as np
import scipy
try:
    import cupy as cp
    import cupyx
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")




class kernel_xPCA():
    """An experimental tool for generating kernel PCA plots
    using a fitted xGPR model. Since this is currently
    experimental, it is not unit-tested. Therefore,
    while it has given useful results in initial
    experiments, it should be used with moderate caution."""

    def __init__(self, dataset, xgpr_model, n_components = 2,
            random_seed = 123):
        """The constructor for kernel_xPCA.

        Args:
            dataset: A valid Dataset object. Should be the same one
                used to train the model preferably, or a subset of it.
            xgpr_model: A trained xGP_Regression object. Hyperparameters
                should already have been tuned and the model should
                already have been fitted. This class does not check
                that model has already been fitted, because if an
                unfitted xGPR model is asked to make predictions it
                will throw an exception.
            n_components (int): The number of principal components to
                obtain. Large numbers will be slower. For a kPCA plot,
                we normally just want 2. For clustering, 100 - 300
                might be more suitable.
            random_seed (int): A seed for the random number generator.
        """
        self.n_components = n_components
        #TODO: It is not good to store the model object here; fix this.
        self.fitted_model = xgpr_model
        dataset.device = self.fitted_model.device
        self.z_mean = self.get_mapped_data_statistics(dataset)
        self.eigvecs = self.initialize_kpca(dataset,
                self.fitted_model.kernel, random_seed)
        dataset.device = "cpu"


    def predict(self, input_x, chunk_size = 2000):
        """Generates a low-dimensional projection of the input."""
        xdata = self.fitted_model.pre_prediction_checks(input_x, get_var = False)
        preds = []
        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            xfeatures = self.fitted_model.kernel.transform_x(xdata[i:cutoff, :])
            xfeatures -= self.z_mean[None,:]
            preds.append(xfeatures @ self.eigvecs)

        if self.fitted_model.device == "gpu":
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
        if self.fitted_model.device == "gpu":
            train_mean = cp.zeros((self.fitted_model.kernel.get_num_rffs()))
        else:
            train_mean = np.zeros((self.fitted_model.kernel.get_num_rffs()))

        for x_data in dataset.get_chunked_x_data():
            if dataset.pretransformed:
                x_features = x_data
            else:
                x_features = self.fitted_model.kernel.transform_x(x_data)
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


    def initialize_kpca(self, dataset, kernel, random_seed):
        """Uses the Nystrom approximation of Z^T Z to find its
        (approximate) top two eigenvectors & eigenvalues.

        Args:
            dataset: An OnlineDataset or OfflineDataset containing the raw data.
            random_seed (int): A seed for the random number generator.

        Returns:
            u_mat (np.ndarray): The eigenvectors of the matrix needed to
                form the preconditioner.
        """
        fitting_rffs = kernel.get_num_rffs()
        rng = np.random.default_rng(random_seed)
        l_mat = rng.standard_normal(size=(fitting_rffs, self.n_components))
        l_mat, _ = np.linalg.qr(l_mat)

        if kernel.device == "cpu":
            acc_results = np.zeros((self.n_components, kernel.get_num_rffs()))
            svd_calculator, cho_calculator = np.linalg.svd, np.linalg.cholesky
            tri_solver = scipy.linalg.solve_triangular
            qr_calculator = np.linalg.qr
        else:
            mempool = cp.get_default_memory_pool()
            acc_results = cp.zeros((self.n_components, kernel.get_num_rffs()))
            svd_calculator, cho_calculator = cp.linalg.svd, cp.linalg.cholesky
            tri_solver = cupyx.scipy.linalg.solve_triangular
            qr_calculator = cp.linalg.qr
            l_mat = cp.asarray(l_mat)

        acc_results = acc_results.T

        self.single_pass_gauss(dataset, kernel, l_mat, acc_results)


        if kernel.device == "gpu":
            mempool.free_all_blocks()

        for i in range(2):
            q_mat, r_mat = qr_calculator(acc_results)
            acc_results[:] = 0.0
            del r_mat
            if kernel.device == "gpu":
                mempool.free_all_blocks()
            self.single_pass_gauss(dataset, kernel, q_mat, acc_results)

        if kernel.device == "gpu":
            mempool.free_all_blocks()

        norm = float( np.sqrt((acc_results**2).sum())  )

        shift = np.spacing(norm)
        acc_results += shift * q_mat
        q_mat = q_mat.T @ acc_results

        q_mat = cho_calculator(q_mat)

        if kernel.device == "gpu":
            mempool.free_all_blocks()

        acc_results = tri_solver(q_mat, acc_results.T,
                            overwrite_b = True, lower=True).T

        u_mat, _, _ = svd_calculator(acc_results, full_matrices=False)
        return u_mat
