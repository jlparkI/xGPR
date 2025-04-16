"""Contains tools to assist in tuning hyperparameters by evaluating
performance on a validation set. This is often best done using
e.g. Optuna but can also be done using optimization
routines in Scipy which are made available here."""
from copy import deepcopy
from scipy.optimize import minimize
import numpy as np
from xGPR import build_classification_dataset
try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import average_precision_score
    import optuna
except:
    pass




def tune_classifier_optuna(training_dataset, validation_dataset,
        classifier, fit_mode="cg", bounds=None,
        eval_metric="cross_entropy",
        max_iter=100, random_seed=123):
    """Tunes a classifier supplied by caller on the validation
    set, fitting it each time to the training set and maximizing
    (or minimizing, as appropriate) a specified metric. Only available
    if Optuna is installed.

    Args:
        training_dataset: A Dataset object for training data that can be
            created by a call to build_classification_dataset (or
            alternatively a custom Dataset object).
        validation_dataset: A Dataset object for validation data that can
            be created by a call to build_classification_dataset (or
            alternatively a custom Dataset object).
        classifier: An xGPDiscriminant object that has 'predict' and 'fit'
            functions available.
        fit_mode (str): One of "cg", "exact". Exact is faster for small
            num_rffs but scales very badly to larger numbers (e.g. > 3000,
            where cg should be preferred.
        bounds: One of None or an np.ndarray. If not None, should have the
            same length as there are number of hyperparameters for the kernel
            of the supplied discriminant. If None, automatically preset
            bounds are used.
        eval_metric (str): One of "cross_entropy", "matthews_corrcoef",
            "accuracy", "aucprc". Determines how performance is evaluated on the
            validation set. For matthews_corrcoef, aucprc and accuracy, maximization
            is performed, while for cross entropy minimization is performed.
        max_iter (int): The maximum number of iterations to run.
        random_seed (int): The random seed used for starting point
            initialization.

    Returns:
        classifier: The updated classifier which has been fitted to the
            data.
        best_score (float): The best score achieved.
        best_hparams (np.ndarray): The best hyperparameters obtained.

    Raises:
        RuntimeError: A RuntimeError is raised if invalid inputs are supplied.
    """
    if eval_metric == "accuracy":
        score_func = accuracy
        sign = -1
    elif eval_metric == "cross_entropy":
        score_func = cross_entropy
        sign = 1
    elif eval_metric == "matthews_corrcoef":
        score_func = matthews_corrcoef
        sign = -1
    elif eval_metric == "aucprc":
        score_func = aucprc
        sign = -1
    else:
        raise RuntimeError("Unknown metric supplied.")

    # Initialize a kernel, then check the bounds.
    classifier.set_hyperparams(dataset=training_dataset)

    if bounds is None:
        bounds = classifier.kernel.get_bounds()
        # The standard bounds for regression for the first hyperparameter
        # are too restrictive for classification; reset them.
        bounds[0,:] = np.array([-11, 1])
        classifier.kernel.set_bounds(bounds)
    else:
        classifier.kernel.set_bounds(bounds)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(lambda trial:
            optuna_loss_func(trial, classifier,
                training_dataset, validation_dataset, score_func,
                bounds, fit_mode, sign),
            n_trials=max_iter)
    best_hparams = [study.best_params[str(i)] for i in range(bounds.shape[0])]

    classifier.set_hyperparams(np.array(best_hparams), training_dataset)
    classifier.fit(training_dataset, mode=fit_mode)

    return classifier, sign * study.best_value, best_hparams





def cv_tune_classifier_optuna(xvalues, yvalues, classifier,
        sequence_lengths = None, fit_mode="cg", bounds=None,
        eval_metric="cross_entropy",
        max_iter=100, random_seed=123):
    """Tunes a classifier supplied by caller using cross-validations.
    This is appropriate for smaller datasets that fit in memory (and
    where the expense of a 5x CV is not excessive). Only available
    if Optuna is installed.

    Args:
        xvalues (np.ndarray): A numpy array of x-values. May be 2d or
            3d; if 3d, sequence lengths must also be supplied.
        yvalues (np.ndarray): A numpy array of y-values.
        classifier: An xGPDiscriminant object that has 'predict' and 'fit'
            functions available.
        sequence_lengths: Either None or a numpy array of shape
            (xvalues.shape[0]). Required if xvalues is a 3d array,
            otherwise ignored. Indicates the length of each sequence
            in xvalues.
        fit_mode (str): One of "cg", "exact". Exact is faster for small
            num_rffs but scales very badly to larger numbers (e.g. > 3000,
            where cg should be preferred.
        bounds: One of None or an np.ndarray. If not None, should have the
            same length as there are number of hyperparameters for the kernel
            of the supplied discriminant. If None, automatically preset
            bounds are used.
        eval_metric (str): One of "cross_entropy", "matthews_corrcoef",
            "accuracy", "aucprc". Determines how performance is evaluated on the
            validation set. For matthews_corrcoef, aucprc and accuracy, maximization
            is performed, while for cross entropy minimization is performed.
        max_iter (int): The maximum number of iterations to run.
        random_seed (int): The random seed used for starting point
            initialization.

    Returns:
        classifier: The updated classifier which has been fitted to the
            data.
        best_score (float): The best score achieved.
        best_hparams (np.ndarray): The best hyperparameters obtained.

    Raises:
        RuntimeError: A RuntimeError is raised if invalid inputs are supplied.
    """
    if eval_metric == "accuracy":
        score_func = accuracy
        sign = -1
    elif eval_metric == "cross_entropy":
        score_func = cross_entropy
        sign = 1
    elif eval_metric == "matthews_corrcoef":
        score_func = matthews_corrcoef
        sign = -1
    elif eval_metric == "aucprc":
        score_func = aucprc
        sign = -1
    else:
        raise RuntimeError("Unknown metric supplied.")

    # Initialize a kernel, then check the bounds.
    training_dataset = build_classification_dataset(xvalues,
            yvalues, sequence_lengths)
    classifier.set_hyperparams(dataset=training_dataset)

    if bounds is None:
        bounds = classifier.kernel.get_bounds()
        # The standard bounds for regression for the first hyperparameter
        # are too restrictive for classification; reset them.
        bounds[0,:] = np.array([-11, 1])
        classifier.kernel.set_bounds(bounds)
    else:
        classifier.kernel.set_bounds(bounds)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(lambda trial:
            optuna_cv_loss_func(trial, classifier,
                xvalues, yvalues, sequence_lengths, score_func,
                bounds, fit_mode, sign),
            n_trials=max_iter)
    best_hparams = [study.best_params[str(i)] for i in range(bounds.shape[0])]

    classifier.set_hyperparams(np.array(best_hparams), training_dataset)
    classifier.fit(training_dataset, mode=fit_mode)

    return classifier, sign * study.best_value, best_hparams





def tune_classifier_powell(training_dataset, validation_dataset,
        classifier, fit_mode="cg", bounds=None,
        eval_metric="cross_entropy", n_restarts=1,
        starting_hparams=None, random_seed=123):
    """Tunes a classifier supplied by caller on the validation
    set, fitting it each time to the training set and maximizing
    (or minimizing, as appropriate) a specified metric.

    Args:
        training_dataset: A Dataset object for training data that can be
            created by a call to build_classification_dataset (or
            alternatively a custom Dataset object).
        validation_dataset: A Dataset object for validation data that can
            be created by a call to build_classification_dataset (or
            alternatively a custom Dataset object).
        classifier: An xGPDiscriminant object that has 'predict' and 'fit'
            functions available.
        fit_mode (str): One of "cg", "exact". Exact is faster for small
            num_rffs but scales very badly to larger numbers (e.g. > 3000,
            where cg should be preferred.
        bounds: One of None or an np.ndarray. If not None, should have the
            same length as there are number of hyperparameters for the kernel
            of the supplied discriminant. If None, automatically preset
            bounds are used.
        eval_metric (str): One of "cross_entropy", "matthews_corrcoef",
            "accuracy", "aucprc". Determines how performance is evaluated on the
            validation set. For matthews_corrcoef, aucprc and accuracy, maximization
            is performed, while for cross entropy minimization is performed.
        n_restarts (int): The number of restarts to use when running the
            optimizer.
        starting_hparams (np.ndarray): If None, the starting hyperparameters are
            set randomly on the first restart. Otherwise, the supplied array
            is used as the first starting point.
        random_seed (int): The random seed used for starting point
            initialization.

    Returns:
        classifier: The updated classifier which has been fitted to the
            data.
        best_score (float): The best score achieved.
        best_hparams (np.ndarray): The best hyperparameters obtained.

    Raises:
        RuntimeError: A RuntimeError is raised if invalid inputs are supplied.
    """
    if eval_metric == "accuracy":
        score_func = accuracy
        sign = -1
    elif eval_metric == "cross_entropy":
        score_func = cross_entropy
        sign = 1
    elif eval_metric == "matthews_corrcoef":
        score_func = matthews_corrcoef
        sign = -1
    elif eval_metric == "aucprc":
        score_func = aucprc
        sign = -1
    else:
        raise RuntimeError("Unknown metric supplied.")

    # Initialize a kernel, then check the bounds.
    classifier.set_hyperparams(dataset=training_dataset)

    if bounds is None:
        bounds = classifier.kernel.get_bounds()
        # The standard bounds for regression for the first hyperparameter
        # are too restrictive for classification; reset them.
        bounds[0,:] = np.array([-11,1])
        classifier.kernel.set_bounds(bounds)
    else:
        classifier.kernel.set_bounds(bounds)

    rng = np.random.default_rng(random_seed)

    best_score = np.inf
    best_hparams = None

    if starting_hparams is None:
        starting_hparams = rng.uniform(low=bounds[:,0], high=bounds[:,1],
                size=bounds.shape[0])

    for _ in range(n_restarts):
        res = minimize(loss_func, starting_hparams,
                args=(classifier, training_dataset, validation_dataset,
                    score_func, fit_mode, sign),
                bounds=list(map(tuple, bounds)),
                method="Powell", options={"xtol":1e-1, "ftol":1e-2})

        if res.fun < best_score:
            best_score = deepcopy(res.fun)
            best_hparams = res.x.copy()

        starting_hparams = rng.uniform(low=bounds[:,0], high=bounds[:,1],
                size=bounds.shape[0])

    classifier.set_hyperparams(best_hparams, training_dataset)
    classifier.fit(training_dataset, mode=fit_mode)

    return classifier, sign * best_score, best_hparams




def loss_func(hyperparams, classifier, training_dataset,
        validation_dataset, score_func, fit_mode="cg", sign=-1):
    """Fits the classifier given a specific set of hyperparameters
    as input and returns a score. The sign on the score is flipped
    if so specified so that maximization can be framed as a
    minimization problem.

    Args:
        hyperparams (np.ndarray): A numpy array of hyperparameters.
        classifier: An object with fit() and predict() functions.
        training_dataset: A Dataset object with training data.
        validation_dataset: A Dataset object with validation data.
        score_func: A function that calculates a metric (accuracy,
            cross entropy etc.) given y values and predictions as
            input.
        fit_mode (str): One of "exact" or "cg". Determines how the
            classifier is fitted.
        sign (int): Either -1 or 1. The output is multiplied by
            this so that maximization can be converted to minimization
            if needed.

    Returns:
        score (float): A value that the scipy routine will try to
            minimize.
    """
    classifier.set_hyperparams(hyperparams, training_dataset)
    classifier.fit(training_dataset, mode=fit_mode)
    y_true, y_pred = [], []

    for x, y, l in validation_dataset.get_chunked_data():
        y_true.append(y)
        y_pred.append(classifier.predict(x, sequence_lengths=l))

    y_true = np.concatenate(y_true)
    y_pred = np.vstack(y_pred)
    score = sign * score_func(y_true, y_pred)
    return score



def optuna_loss_func(trial, classifier, training_dataset,
        validation_dataset, score_func, bounds,
        fit_mode="cg", sign=-1):
    """A modified loss function specific for optuna. Fits the
    classifier given a specific set of hyperparameters as input
    and returns a score. The sign on the score is flipped if so
    specified so that we are always performing minimization for
    simplicity. This is basically a wrapper on lossfunc set up
    to accommodate optuna.
    """
    hparams = [trial.suggest_float(str(i), bounds[i,0],
                bounds[i,1]) for i in range(bounds.shape[0])]

    return loss_func(np.array(hparams), classifier, training_dataset,
            validation_dataset, score_func, fit_mode, sign)


def optuna_cv_loss_func(trial, classifier, xvalues,
        yvalues, sequence_lengths, score_func, bounds,
        fit_mode="cg", sign=-1):
    """A modified loss function specific for cross-validation tuning
    with optuna. Fits the classifier to each split of a 5x CV and
    returns average scores. The sign on the score is flipped if so
    specified so that we are always performing minimization for
    simplicity. This is basically a wrapper on lossfunc set up
    to accommodate optuna.
    """
    kf = KFold(n_splits=5)
    kf.get_n_splits(xvalues)
    results = []
    hparams = [trial.suggest_float(str(i), bounds[i,0],
                bounds[i,1]) for i in range(bounds.shape[0])]

    if len(xvalues.shape) == 2:
        for train_index, test_index in kf.split(xvalues):
            training_dataset = build_classification_dataset(
                    xvalues[train_index,...],
                    yvalues[train_index])
            validation_dataset = build_classification_dataset(
                    xvalues[test_index,...],
                    yvalues[test_index])

            results.append(loss_func(np.array(hparams),
                classifier, training_dataset, validation_dataset,
                score_func, fit_mode, sign))

    elif sequence_lengths is not None and len(xvalues.shape) == 3:
        for train_index, test_index in kf.split(xvalues):
            training_dataset = build_classification_dataset(
                    xvalues[train_index,...],
                    yvalues[train_index],
                    sequence_lengths[train_index])
            validation_dataset = build_classification_dataset(
                    xvalues[test_index,...],
                    yvalues[test_index],
                    sequence_lengths[test_index])

            results.append(loss_func(np.array(hparams),
                classifier, training_dataset, validation_dataset,
                score_func, fit_mode, sign))

    else:
        raise RuntimeError("Sequence lengths must be supplied if "
                "there is a 3d input array.")

    return np.mean(results)

def accuracy(y_true, y_pred):
    """Calculates the accuracy of the predictions.

    Args:
        y_true (np.ndarray): The class labels, of shape (n_samples).
        y_pred (np.ndarray): The predicted probabilities, of shape
            (n_samples, n_classes).

    Returns:
        acc (float): A value between 0 and 1.
    """
    pred_labels = np.argmax(y_pred, axis=1)
    acc = float((pred_labels == y_true).sum())
    return acc / float(y_true.shape[0])


def aucprc(y_true, y_pred):
    """Calculates the AUC-PRC of the predictions.

    Args:
        y_true (np.ndarray): The class labels, of shape (n_samples).
        y_pred (np.ndarray): The predicted probabilities, of shape
            (n_samples, n_classes).

    Returns:
        acc (float): A value between 0 and 1.
    """
    return average_precision_score(y_true, y_pred[:,1])


def cross_entropy(y_true, y_pred):
    """Cross-entropy loss.

    Args:
        y_true (np.ndarray): Ground truth (correct) labels, shape (n_samples).
        y_pred (np.ndarray): array-like of float, shape (n_samples, n_classes).

    Returns:
        loss (float): The cross entropy loss.
    """
    binarized_labels = np.zeros((y_true.shape[0], y_pred.shape[1]))
    # This could be made faster. TODO: Update this
    for i in range(y_pred.shape[1]):
        binarized_labels[y_true==i, i] = 1

    # Note that this introduces some slight error for numbers
    # very close to zero or 1.
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -(binarized_labels * np.log(y_pred)).sum(axis=1)

    return float(np.mean(loss))



def matthews_corrcoef(y_true, y_pred):
    """Compute the Matthews correlation coefficient (MCC).

    Args:
        y_true (np.ndarray): Ground truth labels of shape (n_samples).
        y_pred (np.ndarray): Predicted probabilities of shape (n_samples,
            n_classes).

    Returns
        mcc (float): The Matthews correlation coefficient (+1 represents a perfect
            prediction, 0 an average random prediction and -1 and inverse
            prediction).
    """
    pred_labels = np.argmax(y_pred, axis=1)
    c_mat = np.zeros((y_pred.shape[1], y_pred.shape[1]))

    for yt, yp in zip(y_true.tolist(), pred_labels.tolist()):
        c_mat[yt][yp] += 1

    t_sum = c_mat.sum(axis=1, dtype=np.float64)
    p_sum = c_mat.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(c_mat, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)

    cov_ypyp_ytyt = cov_ypyp * cov_ytyt
    if cov_ypyp_ytyt == 0:
        return 0.0
    return float(cov_ytyp / np.sqrt(cov_ypyp_ytyt))
