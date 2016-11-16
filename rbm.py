import numpy as np
from itertools import product


class RBM:

    def __init__(self, num_visible, num_hidden, learning_rate=0.1,
                 momentum_rate=0.0, regul_param=0.0, regul_bias=False,
                 dropout_p=0.0):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.regul_param = regul_param
        self.regul_bias = regul_bias
        self.dropout_p = 1.0 - dropout_p

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden),
        # using a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.01 * \
            np.random.randn(self.num_visible, self.num_hidden)
        # Insert weights for the bias units into the first row and first
        # column.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    def train(self, data, max_epochs=1000, batch_size=10):
        """
        Train the machine.

        Parameters
        ----------
        data: A matrix where each row is a training example consisting of
        the states of visible units.
        max_epochs: Number of sweeps over the data (default=1000)
        batch_size: Size of the mini-batch, set to None for using all the data
        at each step (default=10)
        """

        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        # Regularization mask
        regul_mask = np.ones(
            (self.num_visible + 1, self.num_hidden + 1), dtype=bool)
        if not self.regul_bias:
            regul_mask[0] = False
            regul_mask[:, 0] = False

        if batch_size is not None:
            batch_nb = 1 + (num_examples - 1) // batch_size
            # Add examples to make data size a multiple of batch_size
            batch_idx = np.random.permutation(
                np.concatenate(
                    [np.arange(num_examples),
                     np.random.choice(num_examples,
                                      batch_size * batch_nb - num_examples,
                                      replace=False)])
            ).reshape(batch_nb, batch_size)
        else:
            batch_size = num_examples
            batch_nb = 1

        # Previous step delta
        delta_weights = 0

        for epoch, batch_id in product(range(max_epochs), range(batch_nb)):
            batch = data if batch_nb == 1 else data[batch_idx[batch_id]]
            # Generate weights with dropouts
            weights = self.weights if self.dropout_p == 1.0 else self.weights \
                * (np.repeat([1, self.dropout_p], [1, self.num_hidden])
                   > np.random.rand(self.num_hidden + 1))
            # Clamp to the batch and sample from the hidden units.
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(batch, weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:, 0] = 1  # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(
                batch_size, self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the
            # hidden states, not the hidden states themselves, when computing
            # associations. We could also use the states; see section 3 of
            # Hinton's "A Practical Guide to Training Restricted Boltzmann
            # Machines" for more.
            pos_associations = np.dot(batch.T, pos_hidden_probs)

            # Reconstruct the visible units and sample again from the hidden
            # units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = np.dot(pos_hidden_states, weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            neg_hidden_probs[:, 0] = 1  # Fix the bias unit.
            # Note, again, that we're using the activation *probabilities*
            # when computing associations, not the states themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Compute step
            delta_weights = self.learning_rate * \
                ((pos_associations - neg_associations) / batch_size) \
                + self.momentum_rate * delta_weights \
                - self.regul_param * (weights * regul_mask)
            # Update weights.
            self.weights += delta_weights

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network
        have been learned), run the network on a set of visible units, to get
        a sample of the hidden units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the
        visible units.

        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden
        units activated from the visible units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the hidden units.
        hidden_activations = self.dropout_p * np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        return hidden_probs[:, 1:] > np.random.rand(
            num_examples, self.num_hidden)

    def run_hidden(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network
        have been learned), run the network on a set of hidden units, to get
        a sample of the visible units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden
        units.

        Returns
        -------
        visible_states: A matrix where each row consists of the visible units
        activated from the hidden units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.weights.T *
                                     np.repeat([1, self.dropout_p],
                                               [1, self.num_hidden])[:, None])
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self._logistic(visible_activations)
        # Turn the visible units on with their specified probabilities.
        return visible_probs[:, 1:] > np.random.rand(
            num_examples, self.num_visible)

    def daydream(self, num_samples):
        """
        Randomly initialize the visible units once, and start
        running alternating Gibbs sampling steps (where each step consists of
        updating all the hidden units, and then updating all of the visible
        units), taking a sample of the visible units at each step. Note that
        we only initialize the network *once*, so these samples are correlated.

        Returns
        -------
        samples: A matrix, where each row is a sample of the visible units
        produced while the network was daydreaming.
        """

        # Create a matrix, where each row is to be a sample of of the visible
        # units (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0, 1:] = np.random.rand(self.num_visible)

        # Taking dropouts in account
        weights = self.weights * \
            np.repeat([1, self.dropout_p], [1, self.num_hidden])

        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i - 1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(
                self.num_visible + 1)
            # Don't overwrite the bias unit in samples
            samples[i, 1:] = visible_states[1:]

        # Ignore the bias units (the first column), since they're
        # always set to 1.
        return samples[:, 1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))
