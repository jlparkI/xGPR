"""Describes the MinibatchDataset class for containing minibatches passed
to model cost functions during adam / stochastic gradient descent."""

class MinibatchDataset:
    """This class enables the adam / stochastic grad descent
    routine to pass an object that has the same necessary
    methods as an OnlineDataset or OfflineDataset to the
    cost functions of a model class -- but this 'dataset'
    contains only a minibatch of data. It lacks the
    methods other dataset classes have for things
    like resetting the device, generating minibatches
    etc. because it will never need to do these things.

    Attributes:
        ndatapoints (int): The total number of datapoints in the
            minibatch.
        xdata: A cupy or numpy array containing the raw x-values
            for the minibatch.
        ydata: A cupy or numpy array containing the raw y-values.
        pretransformed (bool): If True,
            the x-data is already pregenerated random features.
    """

    def __init__(self, xdata, ydata,
                pretransformed = False):
        """MinibatchDataset constructor.

        Args:
            xdata: A cupy or numpy array containing the minibatch
                of data to dispatch.
            ydata: A cupy or numpy array containing the y-values
                associated with this minibatch.
            pretransformed (bool): If True, xdata has already been converted
                to random features.
        """
        self.ndatapoints = xdata.shape[0]
        self.xdata = xdata
        self.ydata = ydata
        self.pretransformed = pretransformed

    def get_chunked_data(self):
        """Returns the minibatch of data to caller."""
        yield self.xdata, self.ydata

    def get_chunked_x_data(self):
        """Returns just the xdata to caller."""
        yield self.xdata

    def get_ndatapoints(self):
        """Returns the number of datapoints."""
        return self.ndatapoints

    def get_pretransformed_status(self):
        """Returns the pretransformed status."""
        return self.pretransformed

    def get_all_data(self):
        """Returns all data at once (without using the generator
        format."""
        return self.xdata, self.ydata
