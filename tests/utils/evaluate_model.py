"""Evaluates a tuned, fitted model on an existing
dataset where expected performance is known."""
from scipy.stats import spearmanr



def evaluate_model(model, train_dataset, test_dataset, get_var = True):
    """Check how well model predictions align with gt, using
    an existing dataset where expected performance is 'known'."""
    if not get_var:
        preds = model.predict(test_dataset._xdata,
                test_dataset._sequence_lengths, get_var)
    else:
        preds, _ = model.predict(test_dataset._xdata,
                test_dataset._sequence_lengths, get_var)
    y_test = test_dataset._ydata * train_dataset._trainy_std
    y_test += train_dataset._trainy_mean
    return spearmanr(preds, y_test)[0]
