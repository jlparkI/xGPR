"""Provides tools for scoring a sample set of hyperparameters on
a cross-validation OR on a validation set using a supplied model."""
import numpy as np

from xGPR.data_handling.offline_data_handling import OfflineDataset
from xGPR.data_handling.dataset_builder import build_online_dataset
from xGPR.data_handling.dataset_builder import build_offline_fixed_vector_dataset
from xGPR.data_handling.dataset_builder import build_offline_sequence_dataset


def _get_cv_score(hparams, model, train_dset, validation_dset,
                random_state, score_type = "mae",
                verbose = True, pretransform_dir = None,
                mode = "cg", cg_tol = 1e-6):
    """Gets a cross-validation score OR a score on a validation dataset
    using the proposed set of hyperparameters.

    Args:
        hparams (np.ndarray): The new set of hyperparameters at which the score
            is evaluated.
        model: A valid xGP_Regression model object.
        train_dset: Either an OnlineDataset or OfflineDataset containing raw
            data.
        validation_dset: Either None or a valid OnlineDataset or OfflineDataset.
            If None, 5x cross-validations are conducted on train_dset. Otherwise,
            the scoring is done on validation_dset, train_dset is used for fitting
            only.
        random_state (int): A seet to the random number generator.
        score_type (str): One of 'mae', 'mse'. If 'mae', mean absolute error
            is optimized. If mse, the mean square error is optimized.
        verbose (bool): If True, regular updates are printed.
        pretransform_dir (str): Either None or a valid filepath where "pretransformed"
            data can be saved. This may provide some speedup on an SSD.
        mode (str): One of "cg", "exact". If "exact", exact fitting is used so no
            preconditioner construction is required. "exact" will raise an error
            if the number of fitting RFFs is > the allowed for the selected kernel.
        cg_tol (float): The tolerance for conjugate gradients convergence.

    Raises:
        ValueError: A ValueError is raised if the inputs are invalid.

    Returns:
        mean_cv_score (float): The mean cross-validation score or validation score
            for the input hparams.
    """
    print(f"******Scoring proposed hyperparams****\n{hparams}", flush=True)
    conv_kernel = False
    if len(train_dset.get_xdim()) == 3:
        conv_kernel = True

    rng = np.random.default_rng(random_state)
    if validation_dset is None:
        if isinstance(train_dset, OfflineDataset):
            if len(train_dset.get_xfiles()) < 5:
                raise ValueError("There must be at least 5 'chunks' to perform "
                    "CV for an offline dataset.")
            idx = rng.permutation(len(train_dset.get_xfiles()))
        else:
            idx = rng.permutation(train_dset.get_ndatapoints())
        idx = np.array_split(idx, 5)
    else:
        idx = [0]

    cv_scores = []
    for i, test_idx in enumerate(idx):
        if validation_dset is not None:
            eval_dset = validation_dset
            fitting_dset = train_dset
        elif isinstance(train_dset, OfflineDataset):
            train_idx = np.concatenate(idx[:i] + idx[i+1:])
            xfiles, yfiles = train_dset.get_xfiles(), train_dset.get_yfiles()
            train_xfiles = [xfiles[j] for j in train_idx.tolist()]
            train_yfiles = [yfiles[j] for j in train_idx.tolist()]

            test_xfiles = [xfiles[j] for j in test_idx.tolist()]
            test_yfiles = [yfiles[j] for j in test_idx.tolist()]

            if conv_kernel:
                fitting_dset = build_offline_sequence_dataset(train_xfiles, train_yfiles,
                                chunk_size = train_dset.get_chunk_size(),
                                skip_safety_checks = True)
                eval_dset = build_offline_sequence_dataset(test_xfiles, test_yfiles,
                                chunk_size = train_dset.get_chunk_size(),
                                skip_safety_checks = True)
            else:
                fitting_dset = build_offline_fixed_vector_dataset(train_xfiles, train_yfiles,
                                chunk_size = train_dset.get_chunk_size(),
                                skip_safety_checks = True)
                eval_dset = build_offline_fixed_vector_dataset(test_xfiles, test_yfiles,
                                chunk_size = train_dset.get_chunk_size(),
                                skip_safety_checks = True)
        else:
        #TODO: We access some private class info here, not great -- update
        #this
            train_x, train_y = train_dset.xdata_[train_idx,:], train_dset.ydata_[train_idx]
            test_x, test_y = train_dset.xdata_[test_idx,:], train_dset.ydata_[test_idx]
            fitting_dset = build_online_dataset(train_x, train_y, chunk_size =
                            train_dset.get_chunk_size())
            eval_dset = build_online_dataset(test_x, test_y, chunk_size =
                            train_dset.get_chunk_size())

        if pretransform_dir is not None:
            input_dataset = model.pretransform_data(fitting_dset, pretransform_dir,
                    random_state, hparams)
        else:
            input_dataset = fitting_dset

        if mode == "cg":
            preconditioner, recommended_mode = build_preconditioner(model, input_dataset,
                hparams, random_state)

            model.fit(input_dataset, preset_hyperparams = hparams,
                random_seed = random_state,
                mode = recommended_mode,
                preconditioner = preconditioner,
                tol = cg_tol, suppress_var = True,
                max_iter = 1000)
        else:
            model.fit(fitting_dset, preset_hyperparams = hparams,
                random_seed = random_state, mode = "exact", suppress_var = True)


        predictions, gtruths = [], []
        trainy_mean, trainy_std = eval_dset.get_ymean(), eval_dset.get_ystd()
        for xchunk, ychunk in eval_dset.get_chunked_data():
            predictions.append(model.predict(xchunk, get_var=False))
            gtruths.append(ychunk * trainy_std + trainy_mean)

        predictions = np.concatenate(predictions)
        gtruths = np.concatenate(gtruths)
        if score_type == "mae":
            cv_scores.append(np.mean(np.abs(gtruths - predictions)))
        elif score_type == "mse":
            cv_scores.append(np.mean( (gtruths - predictions)**2 ) )
        if validation_dset is not None:
            break

    if verbose:
        print(cv_scores, flush=True)
    return np.mean(cv_scores)



def build_preconditioner(xgp_model, fitting_dset, preset_hparams,
        random_state = 123):
    """Build a preconditioner.

    Args:
        xgp_model: An xGP_Regression object.
        fitting_dset: An OnlineDataset or OfflineDataset.
        preset_hparams (np.ndarray): A numpy array containing
            the hyperparameters at which we are fitting.
        random_state (int): A seed for the random number generator.

    Returns:
        preconditioner: Either None if preconditioner construction
            was unsuccessful, or a preconditioner object.
        mode (str): Either "cg" if preconditioner construction
            was successful, or "lbfgs".
    def build_preconditioner(self, dataset, max_rank = 512,
                        adaptive_rank_selection = False,
                        preset_hyperparams = None, random_state = 123):
    """
    mode = "srht_2"
    if xgp_model.device == "cpu":
        mode = "srht"
    preconditioner, ratio = xgp_model.build_preconditioner(fitting_dset,
                    max_rank = 1000, preset_hyperparams = preset_hparams,
                    random_state = random_state, method = mode)

    if ratio > 100:
        improved_precond, _ = xgp_model.build_preconditioner(fitting_dset,
                    max_rank = 2000, preset_hyperparams = preset_hparams,
                    random_state = random_state, method = mode)
        preconditioner = improved_precond
    if preconditioner is None:
        mode = "lbfgs"
    else:
        mode = "cg"

    return preconditioner, mode
