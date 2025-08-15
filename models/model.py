import numpy as np
import torch
import torch.nn as tnn

from models.utils import collate_fn
from models import rnn, vocabulary as mv



class Model:
    """
    Implements an RNN model using SMILES.
    """

    _model_type = "Reinvent"
    _version = 1

    def __init__(
        self,
        vocabulary: mv.Vocabulary,
        tokenizer: mv.SMILESTokenizer,
        network_params: dict = None,
        max_sequence_length: int = 256,
        device=torch.device("cpu"),
    ):
        """
        Implements an RNN using either GRU or LSTM.

        :param vocabulary: vocabulary to use
        :param tokenizer: tokenizer to use
        :param meta_data: model meta data
        :param network_params: parameters required to initialize the RNN
        :param max_sequence_length: maximum length of sequence that can be generated
        :param mode: either "training" or "inference"
        :param device: the PyTorch device
        """

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        if not isinstance(network_params, dict):
            network_params = {}

        self.network = rnn.RNN(len(self.vocabulary), **network_params)
        self.network.to(device)
        self.device = device

        self._nll_loss = tnn.NLLLoss(reduction="none")

    @classmethod
    def create_from_dict(cls, save_dict: dict, device: torch.device):
        vocabulary = None

        if isinstance(save_dict["vocabulary"], dict):
            vocabulary = mv.Vocabulary.load_from_dictionary(save_dict["vocabulary"])
        else:
            vocabulary = save_dict["vocabulary"]
        model = cls(
            vocabulary=vocabulary,
            tokenizer=save_dict.get("tokenizer", mv.SMILESTokenizer()),
            network_params=save_dict.get("network_params"),
            max_sequence_length=save_dict["max_sequence_length"],
            device=device,
        )

        model.network.load_state_dict(save_dict["network"])

        return model

    def get_save_dict(self):
        """Return the layout of the save dictionary"""

        save_dict = dict(
            model_type=self._model_type,
            version=self._version,
            metadata=self.meta_data,
            vocabulary=self.vocabulary.get_dictionary(),
            tokenizer=self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            network=self.network.state_dict(),
            network_params=self.network.get_params(),
        )

        return save_dict

    def save(self, file_path: str) -> None:
        """
        Saves the model into a file.

        :param file_path: Path to the model file.
        """

        save_dict = self.get_save_dict()
        torch.save(save_dict, file_path)

    save_to_file = save  # alias for backwards compatibility

    def likelihood_smiles(self, smiles: str, check: bool=False) -> torch.Tensor:
        sequences = []
        valid_smiles = []
        for smile in smiles:
            try:
                token = self.tokenizer.tokenize(smile)
                encode = self.vocabulary.encode(token)
                sequence = torch.tensor(encode, dtype=torch.long)
                sequences.append(sequence)
                valid_smiles.append(smile)
            except KeyError as e:
                valid_smiles.append(None)

        if check == True:
            return valid_smiles
        else:
            padded_sequences = collate_fn(sequences)
            return self.likelihood(padded_sequences)

    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the likelihood of a given sequence. Used in training.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Log likelihood for each example.
        """

        logits, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)

        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    # NOTE: needed for Reinvent TL
    def sample_smiles(self, num: int = 128, batch_size: int = 128):
        """
        Samples n SMILES from the model.  Is this batched because of memory concerns?

        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :return: A list with SMILES and a list of likelihoods.
        """
        import loguru
        # loguru.logger.debug(f"args: num={num}, batch_size={batch_size}")
        batch_sizes = [batch_size for _ in range(num // batch_size)] + [num % batch_size]
        smiles_sampled = []
        likelihoods_sampled = []

        for size in batch_sizes:
            if not size:
                break

            _, smiles, likelihoods = self.sample(batch_size=size)

            smiles_sampled.extend(smiles)
            likelihoods_sampled.append(likelihoods.data.cpu().numpy())

            del likelihoods

        return smiles_sampled, np.concatenate(likelihoods_sampled)

    @torch.no_grad()
    def sample(self, batch_size: int = 128):
        seqs, likelihoods = self._sample(batch_size=batch_size)

        # FIXME: this is potentially unnecessary in some cases
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()
        ]

        return seqs, smiles, likelihoods

    def _sample(self, batch_size: int = 128):
        """Sample a number of sequences from the RNN

        :param batch_size: batch size which is the number of sequences to sample
        :returns: sequences (2D) and associated NLLs (1D)
        """
        # NOTE: the first token never gets added in the loop so initialize with the start token
        sequences = [torch.full((batch_size, 1), self.vocabulary[mv.START_TOKEN], dtype=torch.long)]
        input_vector = torch.full((batch_size,), self.vocabulary[mv.START_TOKEN], dtype=torch.long)
        hidden_state = None
        nlls = torch.zeros(batch_size, device=self.device)

        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)  # 2D
            log_probs = logits.log_softmax(dim=1)  # 2D
            probabilities = logits.softmax(dim=1)  # 2D
            input_vector = torch.multinomial(probabilities, num_samples=1).view(-1)  # 1D
            sequences.append(input_vector.view(-1, 1))
            nlls += self._nll_loss(log_probs, input_vector)

            if input_vector.sum() == 0:
                break

        concat_sequences = torch.cat(sequences, dim=1)

        return concat_sequences.data, nlls

    def get_network_parameters(self):
        """
        Returns the configuration parameters of the network.

        :returns: network parameters of the RNN
        """

        return self.network.parameters()
