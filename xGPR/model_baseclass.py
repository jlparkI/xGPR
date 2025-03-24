"""Describes the ModelBaseclass from which other model classes inherit.
"""
import sys
import copy
try:
    import cupy as cp
    from .preconditioners.cuda_rand_nys_preconditioners import Cuda_RandNysPreconditioner
except:
    pass
import numpy as np
from .preconditioners.rand_nys_preconditioners import CPU_RandNysPreconditioner
from .preconditioners.tuning_preconditioners import RandNysTuningPreconditioner
from .preconditioners.rand_nys_constructors import srht_ratio_check

from .kernels import KERNEL_NAME_TO_CLASS
from .constants import constants



class ModelBaseclass():
    """The base class for xGPR model classes. Provides shared
    methods and attributes.

    Attributes:
        kernel_choice (str): The kernel selected by the user.
        kernel: The kernel object for the posterior predictive mean. Set to
            None initially then initialized as soon as the user sets hyperparams /
            runs a tuning routine / fits the model.
        weights: An array, either a cp.ndarray or np.ndarray depending on the
            device specified by the user (cpu or cuda). The random features
            generated by self.kernel are multiplied against the weights to
            generate predictions. The weights are calculated during fitting.
        var: A 2d square array, either a cp.ndarray or np.ndarray depending on
            the device specified by the user (cpu or cuda), or a preconditioner
            object for certain kernels (Linear). The random features
            are used in conjunction with var to generate the posterior predictive
            variance. The var is calculated during fitting. Used by regression
            classes only.
        device (str): One of "cuda", "cpu". The user can update this as desired.
            All predict / tune / fit operations are carried out using the
            current device. If there are multiple GPUs, the one currently
            active is used. You can set the currently active GPU by setting
            an environment variable -- e.g. "export CUDA_VISIBLE_DEVICES=1".
        num_rffs (int): The number of random Fourier features used.
        variance_rffs (int): The number of random Fourier features used for
            calculating posterior predictive variance.
        kernel_settings (dict): Contains kernel-specific parameters --
            e.g. 'matern_nu' for the nu for the Matern kernel, or 'conv_width'
            for the conv1d kernel.
        verbose (bool): If True, regular updates are printed during
            hyperparameter tuning and fitting.
        num_threads (int): The number of threads to use for random feature generation
            if running on CPU. If running on GPU, this argument is ignored.
        double_precision_fht (bool): If True, use double precision during FHT for
            generating random features. For most problems, it is not beneficial
            to set this to True -- it merely increases computational expense
            with negligible benefit -- but this option is useful for testing.
        exact_var_calculation (bool): If True, variance is calculated exactly (within
            the limits of the random feature approximation). If False, a preconditioner
            is used. The preconditioner approach is only used for linear kernels.
        random_seed (int): The seed to the random number generator. Used throughout the
            model class whenever randomness is desired. Can be reset by user.
            Resetting the random seed -- like changing the number of rffs -- will
            cause the kernel to be re-initialized.
        n_classes (int): The number of classes expected in the
            data. This is initialized when fit is called. Used for classification
            only.
        _gamma (ndarray): Either None (if model has not been fitted or is regression)
            or an array of shape (n_classes). Used for classification only.
    """

    def __init__(self, num_rffs:int = 256, variance_rffs:int = 16,
            kernel_choice:str = "RBF", device:str = "cpu",
            kernel_settings:dict = constants.DEFAULT_KERNEL_SPEC_PARMS,
            verbose:bool = True,
            num_threads:int = 2,
            random_seed:int = 123) -> None:
        """Constructor.

        Args:
            num_rffs (int): The number of random Fourier features
                to use. For certain kernels (Linear) this will be set
                by the data.
            variance_rffs (int): The number of random Fourier features
                to use for posterior predictive variance (i.e. calculating
                uncertainty on predictions). Defaults to 64.
            kernel_choice (str): The kernel that the model will use.
                Defaults to 'RBF'. Must be in kernels.kernel_list.
                KERNEL_NAME_TO_CLASS.
            device (str): Determines whether calculations are performed on
                'cpu' or 'cuda'. The initial entry can be changed later
                (i.e. model can be transferred to a different device).
                Defaults to 'cpu'.
            kernel_settings (dict): Contains kernel-specific parameters --
                e.g. 'matern_nu' for the nu for the Matern kernel, or 'conv_width'
                for the conv1d kernel.
            verbose (bool): If True, regular updates are printed
                during fitting and tuning. Defaults to True.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. If running on GPU, this argument is ignored.
            random_seed (int): The seed to the random number generator.
        """
        self.kernel_choice = kernel_choice
        self.kernel = None
        self.weights = None
        #Classification classes don't use var or trainy_mean, trainy_std
        self.var = None
        self.trainy_mean = 0
        self.trainy_std = 1

        self.device = device

        self.num_rffs = num_rffs
        #Variance_rffs must be <= num_rffs. Always set second
        self.variance_rffs = variance_rffs

        self.kernel_spec_parms = kernel_settings
        self.num_threads = num_threads

        self.verbose = verbose

        #Currently we do not allow user to set double_precision_fht --
        #only used for testing.
        self.double_precision_fht = False
        # Classification classes don't use exact_var
        self.exact_var_calculation = True
        self.random_seed = random_seed

        #Regression classes don't use n_classes or gamma.
        self.n_classes = 1
        self._gamma = None


    def pre_prediction_checks(self, input_x, sequence_lengths, get_var:bool):
        """Checks input data to ensure validity.

        Args:
            input_x (np.ndarray): A numpy array containing the input data.
            sequence_lengths: None if you are using a fixed-vector kernel (e.g.
                RBF) and a 1d array of the number of elements in each sequence /
                nodes in each graph if you are using a graph or Conv1d kernel.
            get_var (bool): Whether a variance calculation is desired.

        Returns:
            x_array: A cupy array (if self.device is cuda) or a reference
                to the unmodified input array otherwise.

        Raises:
            RuntimeError: If invalid inputs are supplied,
                a detailed RuntimeError is raised to explain.
        """
        if self.kernel is None or self.weights is None:
            raise RuntimeError("Model has not yet been successfully fitted.")
        if not self.kernel.validate_new_datapoints(input_x):
            raise RuntimeError("The input has incorrect dimensionality.")
        if sequence_lengths is None:
            if len(input_x.shape) != 2:
                raise RuntimeError("sequence_lengths is required if using a "
                        "convolution kernel.")
        else:
            if len(input_x.shape) == 2:
                raise RuntimeError("sequence_lengths must be None if using a "
                    "fixed vector kernel.")

        #This should never happen, but just in case.
        if self.weights.shape[0] != self.kernel.get_num_rffs():
            raise RuntimeError("The size of the weight vector does not "
                    "match the number of random features that are generated.")
        if self.var is None and get_var:
            raise RuntimeError("Variance was requested but suppress_var "
                    "was selected when fitting, meaning that variance "
                    "has not been generated.")
        if self.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()


    def set_hyperparams(self, hyperparams = None, dataset = None):
        """Sets the hyperparameters to those supplied if
        the kernel already exists, creates a kernel if it
        doesn't already exist. If the kernel doesn't already
        exist a dataset needs to be supplied to tell the
        kernel what size its inputs will be. If hyperparams is None,
        the kernel is initialized.

        Args:
            hyperparams (ndarray): A numpy array such that shape[0] == the number of
                hyperparameters for the selected kernel.
            dataset: Either None or a valid dataset object. If the kernel already
                exists (i.e. if set hyperparams or any of the tuning / fitting /
                scoring routines has already been called) no dataset is required
                and this argument is ignored. If the kernel does NOT exist,
                this argument is required.

        Raises:
            RuntimeError: A RuntimeError is raised if the kernel does not exist but
                a dataset was not supplied.
        """
        if self.kernel is None and dataset is None:
            raise RuntimeError("A dataset is required if the kernel has not already "
                    "been initialized. The kernel is initialized by calling set_hyperparams "
                    "or fit or any of the hyperparameter tuning / scoring routines.")
        if hyperparams is None and dataset is None:
            raise RuntimeError("Should supply hyperparams and/or a dataset.")
        if self.kernel is None:
            self._initialize_kernel(dataset, hyperparams = hyperparams)
        else:
            self.kernel.check_hyperparams(hyperparams)
            self.kernel.set_hyperparams(hyperparams, logspace = True)
            self.weights = None
            self.gamma = None
            self.var = None


    def get_hyperparams(self):
        """Simple helper function to return hyperparameters if the model
        has already been tuned or fitted."""
        if self.kernel is None:
            return None
        return self.kernel.get_hyperparams()



    def _initialize_kernel(self, dataset = None, xdim = None,
            hyperparams = None, bounds = None):
        """Selects and initializes an appropriate kernel object based on the
        kernel_choice string supplied by caller. The kernel is then moved to
        the appropriate device based on the 'device' supplied by caller
        and is returned.

        Args:
            dataset: Either None or a valid dataset object. If None, xdim must not be
                None.
            xdim: Either None or the numpy array returned by dataset.get_xdim(). If None,
                dataset must not be None.
            hyperparams (ndarray): Either None or a numpy array. If not None,
                must be a numpy array such that shape[0] == the number of hyperparameters
                for the selected kernel. The kernel hyperparameters are then initialized
                to the specified value. If None, default hyperparameters are used which
                should then be tuned.
            bounds (np.ndarray): The bounds on hyperparameter
                tuning. Must have an appropriate shape for the
                selected kernel. If None, the kernel will use
                its defaults. Defaults to None.

        Returns:
            kernel: An object of the appropriate kernel class.

        Raises:
            RuntimeError: Raises a value error if an unrecognized kernel
                is supplied.
        """
        if self.kernel_choice not in KERNEL_NAME_TO_CLASS:
            raise RuntimeError("An unrecognized kernel choice was supplied.")

        if dataset is None and xdim is not None:
            input_xdim = xdim
        elif dataset is not None:
            input_xdim = dataset.get_xdim()
        else:
            raise RuntimeError("Either a dataset or xdim must be supplied.")

        self.kernel = KERNEL_NAME_TO_CLASS[self.kernel_choice](input_xdim,
                            self.num_rffs, self.random_seed, self.device,
                            self.num_threads, self.double_precision_fht,
                            kernel_spec_parms = self.kernel_spec_parms)

        #Some kernels set the number of rffs themselves (Linear). If so,
        #reset the number of RFFs for the class to ensure it matches
        #what the kernel is using. We avoid calling the setter and set
        #the attribute directly since the setter may try to reinitialize
        #the kernel under some circumstances.
        self._num_rffs = self.kernel.get_num_rffs()
        if self.variance_rffs >= self.num_rffs:
            raise RuntimeError("The number of variance rffs must be < num_rffs.")

        if bounds is not None:
            self.kernel.set_bounds(bounds)
        if hyperparams is not None:
            self.kernel.check_hyperparams(hyperparams)
            self.kernel.set_hyperparams(hyperparams, logspace = True)
        self.weights, self.var = None, None


    def _run_pre_nmll_prep(self, dataset, bounds = None):
        """Runs key steps / checks needed if about to calculate
        NMLL.
        """
        if self.kernel is None:
            self._initialize_kernel(dataset, bounds = bounds)
        self.weights, self.var = None, None
        return self.kernel.get_bounds()


    def _run_singlepoint_nmll_prep(self, dataset, exact_method:bool = False):
        """Runs key steps / checks needed if about to calculate
        NMLL at a single point, in which case bounds are not needed.
        """
        if self.kernel is None:
            self._initialize_kernel(dataset)

        self.weights, self.var = None, None

        if self.num_rffs <= 2:
            raise RuntimeError("num_rffs should be > 2 to use any tuning method.")

        if exact_method:
            if self.kernel.get_num_rffs() > constants.MAX_CLOSED_FORM_RFFS:
                raise RuntimeError(f"At most {constants.MAX_CLOSED_FORM_RFFS} can be used "
                        "for tuning hyperparameters using this method. Try tuning "
                        "using approximate nmll instead.")



    def _run_pre_fitting_prep(self, dataset, max_rank = None):
        """Runs key steps / checks needed if about to fit the
        model.
        """
        self.trainy_mean = dataset.get_ymean()
        self.trainy_std = dataset.get_ystd()

        if self.kernel is None:
            self._initialize_kernel(dataset)
        if self.variance_rffs > self.kernel.get_num_rffs():
            raise RuntimeError("The number of variance rffs should be <= the number "
                    "of random features for the kernel.")
        if max_rank is not None:
            if max_rank < 1:
                raise RuntimeError("Invalid value for max_rank.")
            if max_rank >= self.kernel.get_num_rffs():
                raise RuntimeError("Max rank should be < the number of rffs.")




    def _check_rank_ratio(self, dataset, sample_frac:float = 0.1,
            max_rank:int = 512):
        """Determines what ratio a particular max_rank would achieve by using a random
        sample of the data. This can be used to determine if a particular max_rank is
        'good enough' to achieve a fast fit, and if so, that max_rank can be used
        for fitting. If the number of rffs is > 8192, this function will use
        8192 by default for better speed (we can get away with this thanks to
        eigenvalue interlacing on random matrices). Note that this procedure
        is not advisable for a small number of datapoints (e.g. < 5,000);
        for those cases, just building the full preconditioner is easier.

        Args:
            dataset: A Dataset object.
            max_rank (int): The maximum rank for the preconditioner, which
                uses a low-rank approximation to the matrix inverse. Larger
                numbers mean a more accurate approximation and thus reduce
                the number of iterations, but make the preconditioner more
                expensive to construct.
            sample_frac (float): The fraction of the data to sample. Must be in
                [0.01, 1].

        Returns:
            achieved_ratio (float): The min eigval of the preconditioner over
                lambda, the noise hyperparameter shared between all kernels.
                This value has decent predictive value for assessing how
                well a preconditioner built with this max_rank is likely to perform.
        """
        if sample_frac < 0.01 or sample_frac > 1:
            raise RuntimeError("sample_frac must be in the range [0.01, 1]")

        reset_num_rffs = False
        if self.num_rffs > 8192:
            reset_num_rffs, num_rffs = True, copy.deepcopy(self.num_rffs)
            if self.kernel_choice == "RBFLinear" and self.num_rffs % 2 != 0:
                self.num_rffs = 8191
            else:
                self.num_rffs = 8192

        s_mat = srht_ratio_check(dataset, max_rank, self.kernel, self.random_seed,
                self.verbose, sample_frac)
        ratio = float(s_mat.min() / self.kernel.get_lambda()**2) / sample_frac

        if reset_num_rffs:
            self.num_rffs = num_rffs

        return ratio



    def _autoselect_preconditioner(self, dataset, min_rank:int = 512,
            max_rank:int = 3000, increment_size:int = 512,
            always_use_srht2:bool = False,
            ratio_target:float = 30., tuning:bool = False):
        """Uses an automated algorithm to choose a preconditioner that
        is up to max_rank in size. For internal use only.

        Args:
            dataset: A Dataset object.
            max_rank (int): The largest rank that should be used for
                building the preconditioner.
            increment_size (int): The amount by which to increase the
                preconditioner size when increasing the max rank.
            always_use_srht2 (bool): If True, srht2 should always be used
                regardless of the ratio obtained. This will reduce the
                number of iterations about 30% but increase time cost
                of preconditioner construction about 150%.
            ratio_target (int): The target value for the ratio.
            tuning (bool): If True, preconditioner is intended for tuning,
                so it also needs to be used for SLQ.

        Returns:
            preconditioner: A preconditioner object.
        """
        sample_frac, method, ratio, rank = 0.2, "srht", np.inf, min_rank
        actual_num_rffs = self.kernel.get_num_rffs()

        if rank >= actual_num_rffs:
            rank = actual_num_rffs - 1
            ratio = 0.5 * ratio_target

        if dataset.get_ndatapoints() < 5000:
            sample_frac = 1

        while ratio > ratio_target and rank < max_rank:
            ratio = self._check_rank_ratio(dataset, sample_frac = sample_frac,
                    max_rank = rank)
            if ratio > ratio_target:
                if (rank + increment_size) < max_rank and \
                        (rank + increment_size) < actual_num_rffs:
                    rank += increment_size
                else:
                    rank = max_rank
                    if rank > actual_num_rffs:
                        rank = actual_num_rffs - 1
                    method = "srht_2"
                    break

        if self.verbose:
            print(f"Using rank: {rank}")

        if always_use_srht2:
            method = "srht_2"

        if tuning:
            preconditioner = RandNysTuningPreconditioner(self.kernel, dataset, rank,
                        False, self.random_seed, method)
        else:
            if self.device == "cuda":
                preconditioner = Cuda_RandNysPreconditioner(self.kernel, dataset, rank,
                        self.verbose, self.random_seed, method)
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            else:
                preconditioner = CPU_RandNysPreconditioner(self.kernel, dataset, rank,
                        self.verbose, self.random_seed, method)

        return preconditioner



    ####The remaining functions are all getters / setters.


    @property
    def kernel_spec_parms(self):
        """Property definition for the kernel_spec_parms."""
        return self._kernel_spec_parms

    @kernel_spec_parms.setter
    def kernel_spec_parms(self, value):
        """Setter for kernel_spec_parms. If the
        user is changing this, the kernel needs to be
        re-initialized."""
        if not isinstance(value, dict):
            raise RuntimeError("Tried to set kernel_spec_parms to something that "
                    "was not a dict!")
        self._kernel_spec_parms = value
        self.kernel = None
        self.weights = None
        self.gamma = None
        self.var = None


    @property
    def kernel_choice(self):
        """Property definition for the kernel_choice attribute."""
        return self._kernel_choice

    @kernel_choice.setter
    def kernel_choice(self, value):
        """Setter for the kernel_choice attribute. If
        the user is changing this, the kernel needs to
        be re-initialized."""
        if not isinstance(value, str):
            raise RuntimeError("You supplied a kernel_choice that is not a string.")
        if value not in KERNEL_NAME_TO_CLASS:
            raise RuntimeError("You supplied an unrecognized kernel.")
        self._kernel_choice = value
        self.kernel = None
        self.weights = None
        self.gamma = None
        self.var = None

    @property
    def num_rffs(self):
        """Property definition for the num_rffs attribute."""
        return self._num_rffs

    @num_rffs.setter
    def num_rffs(self, value):
        """Setter for the num_rffs attribute. If the
        user is changing this, the kernel needs to
        be re-initialized."""
        self._num_rffs = value
        if self.kernel is not None:
            self._initialize_kernel(xdim = self.kernel.get_xdim(),
                   hyperparams = self.kernel.get_hyperparams(),
                   bounds = self.kernel.get_bounds())
        self.weights = None
        self.gamma = None
        self.var = None

    @property
    def variance_rffs(self):
        """Property definition for the variance_rffs attribute."""
        return self._variance_rffs

    @variance_rffs.setter
    def variance_rffs(self, value):
        """Setter for the variance_rffs attribute. If the
        user is changing this, the kernel may need to
        be re-initialized, at least if variance has already
        been calculated. Need to be careful here, because
        for certain fitting procedures, the fit() routine
        does reset variance_rffs."""
        if value > constants.MAX_VARIANCE_RFFS:
            raise RuntimeError("Currently to keep computational expense at acceptable "
                    f"levels variance rffs is capped at {constants.MAX_VARIANCE_RFFS}.")
        if value > self.num_rffs and self.kernel_choice not in ["Linear", "ExactQuadratic"]:
            raise RuntimeError("variance_rffs must be <= num_rffs.")
        self._variance_rffs = value
        if self.var is not None:
            if self.kernel is not None:
                self._initialize_kernel(xdim = self.kernel.get_xdim(),
                    hyperparams = self.kernel.get_hyperparams(),
                    bounds = self.kernel.get_bounds())
            self.weights = None
            self.gamma = None
            self.var = None


    @property
    def num_threads(self):
        """Property definition for the num_threads attribute."""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value):
        """Setter for the num_threads attribute."""
        if value < 1:
            self._num_threads = 1
            raise RuntimeError("Num threads if supplied must be an integer > 1.")
        self._num_threads = value
        if self.kernel is not None:
            self.kernel.num_threads = value

    @property
    def double_precision_fht(self):
        """Property definition for the double_precision_fht attribute."""
        return self._double_precision_fht


    @double_precision_fht.setter
    def double_precision_fht(self, value):
        """Setter for the double_precision_fht attribute."""
        self._double_precision_fht = value
        if self.kernel is not None:
            self.kernel.double_precision = value

    @property
    def random_seed(self):
        """Property definition for the random_seed attribute."""
        return self._random_seed


    @random_seed.setter
    def random_seed(self, value):
        """Setter for the random_seed attribute. If this is
        reset the kernel needs to be re-initialized."""
        self._random_seed = value
        if self.kernel is not None:
            self._initialize_kernel(xdim = self.kernel.get_xdim(),
                   hyperparams = self.kernel.get_hyperparams(),
                   bounds = self.kernel.get_bounds())
        self.weights = None
        self.gamma = None
        self.var = None


    @property
    def device(self):
        """Property definition for the device attribute."""
        return self._device

    @device.setter
    def device(self, value):
        """Setter for the device attribute."""
        if value not in ["cpu", "cuda"]:
            raise RuntimeError("Device must be in ['cpu', 'cuda'].")

        if "cupy" not in sys.modules and value == "cuda":
            raise RuntimeError("You have specified cuda mode but CuPy is "
                "not installed. Currently CPU only fitting is available.")

        if "xGPR.xgpr_cuda_rfgen_cpp_ext" not in sys.modules and value == "cuda":
            raise RuntimeError("You have specified the cuda fit mode but the "
                "rfgen cuda module is not installed / "
                "does not appear to have installed correctly. "
                "Currently CPU only fitting is available.")

        if self.kernel is not None:
            self.kernel.device = value
        if self.weights is not None:
            if value == "cuda":
                self.weights = cp.asarray(self.weights)
            elif value == "cpu" and not isinstance(self.weights, np.ndarray):
                self.weights = cp.asnumpy(self.weights)
        if self.var is not None:
            if self.exact_var_calculation:
                if value == "cpu":
                    self.var = cp.asnumpy(self.var)
                else:
                    self.var = cp.asarray(self.var)
            else:
                self.var.device = value
        if value == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        self._device = value
