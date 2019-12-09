import torch.optim

from ..hyperp import Defaults

class StandardOptimizer:
    """Standard Transformer optimizer defined in the paper Attention Is All You Need.

    This uses the Adam optimizer with b1 = 0.9, b2 = 0.98, and eps = 1e-9. Learning rate varies
    according to the following formula defined in the aforementioned paper:

        LR = sqrt( d(E) ) * min( sqrt( step_num ), step_num * warmup_steps^(-1.5) )

    Here d(E) is the dimensionality of the embedding space used by the Transformer, and warmup_steps
    is the number of warmup steps to allow the learning rate to increase before it starts to
    decrease with the inverse square root of the step number.

    This implementation is based heavily on the one provided in The Annotated Transformer here:
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__( self,
        embedding_size=Defaults.EmbeddingSize,
        warmup_steps=Defaults.OptimizerWarmupSteps ):
        self.current_step = 0
        self.current_learning_rate = 0.0
        self.warmup_steps = warmup_steps
        self.embedding_size = embedding_size
        self.optimizer = torch.optim.Adam( lr=0.0, betas=(0.9, 0.98), eps=1e-9 )

    def update_learning_rate( self ):
        """Updates the optimizer's learning rate based on the formula given in the paper Attention
        Is All You Need."""
        self.current_learning_rate = (self.embedding_size ** -0.5 *
            min( self.current_step ** -0.5, self.current_step * self.warmup_steps ** 1.5 ) )

    def step( self ):
        """Computes the new learning rate and runs another step of the optimizer."""
        self.current_step += 1
        self.update_learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_learning_rate
        self.optimizer.step()

    def zero_grad( self ):
        self.optimizer.zero_grad()
