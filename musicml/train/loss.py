import torch
import torch.nn.functional as F

from ..hyperp import Defaults

class LabelSmoothing:
    """Label smoothing loss criterion.

    Adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """
    def __init__( self, vocab_size, smoothing_value=Defaults.LabelSmoothingValue ):
        super().__init__()
        self.confidence = 1.0 - smoothing_value
        self.smoothing_value = smoothing_value
        self.vocab_size = vocab_size

    def __call__( self, input, target ):
        assert input.size( -1 ) == self.vocab_size

        true_distribution = input.data.clone()
        true_distribution.fill_( self.smoothing_value / (self.vocab_size - 2) )
        true_distribution.scatter_( 1, target.data.unsqueeze( 1 ), self.confidence )

        # Expect input to be logits, not log-probabilities.
        return F.kl_div( F.log_softmax( input, dim=-1 ), true_distribution, reduction="batchmean" )
