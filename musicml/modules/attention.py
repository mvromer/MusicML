import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from ..hyperp import Defaults

class MultiheadAttention( nn.Module ):
    """Multihead attention mechanism with optional relative positional embedding.

    Based on the mechanism described in Attention Is All You Need from Vaswani et al. Also
    incorporates a variation of relative positional self-attention described in Huang et al.'s paper
    Music Transformer: Generating Music with Long-term Structure.
    """

    def __init__( self, embedding_size=Defaults.EmbeddingSize,
        number_heads=Defaults.NumberAttentionHeads,
        key_size=None, value_size=None,
        embed_relative_positions=False,
        max_relative_distance=Defaults.MaxRelativeAttentionDistance ):
        """Creates a new multihead attention layer.

        Args:
            embedding_size: Dimensionality of the source/target sequence's embedding vectors.
                Denoted as d(E).
            number_heads: Number of heads attending. Denoted as H.
            key_size: Dimensionality of each key projected. Denoted as d(K). If not given, defaults
                to embedding_size.
            value_size: Dimensionality of each value projected. Denoted as d(V). If not given,
                defaults to embedding_size.
            embed_relative_positions: If True, relative position embeddings will be learned and
                incorporated into the attention weights computed on a per-head basis.
            max_relative_distance: If embed_relative_positions is True, then this is the maximum
                relative distance learned. Denoted as C.
        """
        super().__init__()
        self.number_heads = number_heads
        self.key_size = key_size or embedding_size
        self.value_size = value_size or embedding_size

        # Transformations that project source sequences into key and value spaces. Expressed as a
        # linear layer with no bias.
        self.key_trans = nn.Linear( embedding_size, self.key_size * self.number_heads, bias=False )
        self.value_trans = nn.Linear( embedding_size, self.value_size * self.number_heads, bias=False )

        #  Transformation that projects target sequences into query space. Expressed as a linear
        # layer with no bias.
        self.query_trans = nn.Linear( embedding_size, self.key_size * self.number_heads, bias=False )

        # Scaling factor applied after computing QK^T.
        self.scale_factor = self.key_size ** -0.5

        # Relative position embeddings learned if they are to be used during the attention
        # computation. They are divided into two sets. rpe_neg contains the embeddings for each
        # relative position with a nonpositive distance ordered from -C to 0. rpe_pos contains the
        # embeddings for each relative position with a positive distance ordered from 1 to C. Each
        # embedding has dimensionality d(K) and is learned on a per-head basis.
        #
        # NOTE: Conceptually rpe_neg and rpe_pos could be stored as (C + 1) x d(K) and C x d(K),
        # respectively. However, for certain padding operations used during the forward pass, and
        # given the fact that the tensors resulting from cropping/padding these embedding parameters
        # need to be transposed prior to multiplying them with Q, we store the relative position
        # embeddings transposed as H x d(K) x (C + 1) and H x d(k) x C respectively for rpe_neg and
        # rpe_pos.
        #
        self.embed_relative_positions = embed_relative_positions
        self.max_relative_distance = max_relative_distance

        if self.embed_relative_positions:
            if max_relative_distance <= 0:
                raise ValueError( "Relative position embedding is enabled. Max relative distance must be positive." )

            self.rpe_neg = nn.Parameter( torch.empty( self.number_heads, self.key_size, self.max_relative_distance + 1 ) )
            self.rpe_pos = nn.Parameter( torch.empty( self.number_heads, self.key_size, self.max_relative_distance ) )

            torch.nn.init.xavier_uniform_( self.rpe_neg )
            torch.nn.init.xavier_uniform_( self.rpe_pos )
        else:
            self.rpe_neg = None
            self.rpe_pos = None

        # Softmax layer applied to innermost dimension of Z = (QK^T) / sqrt(d(K)) during attention
        # calculation.
        self.z_softmax = nn.Softmax( dim=-1 )

        # Projection from the matrix of attention values concatenated from each head to our original
        # embedding vector space. Expressed as linear layer with no bias.
        self.z_trans = nn.Linear( self.value_size * self.number_heads, embedding_size, bias=False )

    def forward( self, source, target, attention_mask=None ):
        """Computes multihead attention for the given source and target sequences.

        Each head will project its keys and values from the given source sequence and its queries
        from the given target sequence. The result of this is a tensor with dimensions T x d(E).

        Args:
            source: Source sequence with dimensions S x d(E).
            target: Target sequence with dimensions T x d(E). If this attention mechanism is
                configured to also embed relative positions during the attention weight computation,
                then S must equal T.
            attention_mask: T x S additive mask applied to attention calculation prior to applying
                softmax. If given, entries corresponding to masked values must be set to -inf and
                unmasked values must be set to zero.
        """
        # Project keys, values, and queries from inputs.
        keys = self.key_trans( source )
        values = self.value_trans( source )
        queries = self.query_trans( target )

        # Each output can be viewed as a 1-by-H block matrix containing the keys, queries, and
        # values for each head in the multihead attention mechanism. Each block is sized as follows:
        #
        #   For keys: S x d(K)
        #   For values: S x d(V)
        #   For queries: T x d(K)
        #
        # To perform a batch multiplication, we need to view each as a block tensor with the
        # following size:
        #
        #   For keys: H x d(K) x S (because we need the transpose of each block)
        #   For values: H x S x d(V)
        #   For queries: H x T x d(K)
        #
        # For values and queries, we need transpose the linear layer output, view it as a 3D tensor,
        # and then permute the dimensions. For the keys (since we need the transpose), we only need
        # to tranpose the linear layer output and view it as a 3D tensor.
        #
        keyTransView = keys.t().view( self.number_heads, self.key_size, -1 )
        queriesView = queries.t().view( self.number_heads, self.key_size, -1 ).permute( 0, 2, 1 )
        valuesView = values.t().view( self.number_heads, self.value_size, -1 ).permute( 0, 2, 1 )

        # Start computing attention in parallel across all heads. First Z = QK^T.
        z = torch.bmm( queriesView, keyTransView )

        # If we are embedding relative positions, then compute the relative position logits that
        # will modulate are attention logits in Z.
        if self.embed_relative_positions:
            relative_logits = self.compute_relative_logits( queriesView )
            z = z + relative_logits

        # Scale by root inverse of K.
        z = self.scale_factor * z

        # If given, add the attention mask prior to computing softmax. At this point Z has
        # dimensions H x T x S. Adding the attention mask, which must be T x S, to Z will cause it
        # to be broadcast across all heads, which is what we want.
        if attention_mask is not None:
            z = z + attention_mask

        # Softmax along innermost dimension of Z.
        z = self.z_softmax( z )

        # Finally multiply by V to get final attention value for each head.
        z = torch.bmm( z, valuesView )

        # Concatenate each head's attention value horizontally across Z to make it T x (d(V) * H).
        z = torch.cat( torch.unbind( z, dim=0 ), dim=1 )

        # Project the final multihead attention value back into our original embedding vector space.
        return self.z_trans( z )

    def compute_relative_logits( self, queries ):
        """Computes the relative positional logits to embed with the attention logits."""
        # Let L = L_l + L_u be the relative logits we want to compute for a given head, where L_l is
        # the lower triangular portion of L including the main diagonal, and L_u is the upper
        # triangular portion above the main diagonal.
        #
        # We define L_l = skew_lower( Q(R_l)^T ) and L_u = skew_upper( Q(R_u)^T ). Huang et al.
        # gives an efficient algorithm for computing L_l. The skew_lower function matches the skew
        # procedure given in Huang et al. with the added step of zeroing out the upper triangular
        # portion of the result.
        #
        # The matrix R_l is the T x d(K) matrix defined in Huang et al. containing the relative
        # position embeddings that are applied with the query matrix Q, which also has dimensions of
        # T x d(K). The rows 0 to (T-1) of R_l correspond to relative distances (T-1) to 0. These
        # relative distances then map to the corresponding relative position embedding.
        #
        # As per Shaw et al., relative distances exceeding the max relative distance C configured
        # for the relative attention mechanism are clipped to -C and C. Thus, the relative attention
        # mechanism only learns 2C + 1 relative position embeddings per head as part of its model
        # parameters.
        #
        # If T = C+1, then constructing R_l is trivial because it uses precisely all of the relative
        # position embeddings corresponding to negative relative distances. If T < C+1, then we only
        # need to populate R_l with the position embeddings corresponding to relative distances T-1
        # to 0. If T > C+1, then we populate the last T rows with relative position embeddings for
        # distances -C to 0 and replicate the position embedding for distance -C across the first
        # T-C-1 rows of R_l.
        #
        # In practice, we construct (R_l)^T directly rather than R_l since that is the matrix we
        # ultimately need for computing L_l. Thus, within the code, all cropping and padding occurs
        # across columns rather than across rows.
        #
        # The result of Q(R_l)^T is passed through the skew_lower function, which returns L_l.
        #
        # A similar procedure is done to construct R_u. Then, the result of Q(R_u)^T is passed
        # through skew_upper, which works in principle similar to skew_lower. The only difference is
        # that the padding column is added to the right instead of the left as is done in Huang et
        # al. for the skew_lower function. Also, skew_upper zeros out the lower triangular portion
        # AND the main diagonal of the final result. This final value is L_u.
        #
        L_l = self.compute_relative_logits_lower( queries )
        L_u = self.compute_relative_logits_upper( queries )
        return L_l + L_u

    def compute_relative_logits_lower( self, queries ):
        """Computes the lower triangular portion of the relative positional logits."""
        T = queries.size( dim=-2 )

        if T <= (self.max_relative_distance + 1):
            # Crop out the negative relative position embeddings we need to build (R_l)^T.
            R_l_trans = self.rpe_neg[:, :, -T:]
        else:
            # Need to pad out the negative relative position embeddings, replicating the embedding
            # for distance -C across the first T-C-1 columns of (R_l)^T.
            pad_amount = T - self.max_relative_distance - 1
            R_l_trans = F.pad( self.rpe_neg, (pad_amount, 0), mode="replicate" )

        return self.skew_lower( torch.bmm( queries, R_l_trans ) )

    def skew_lower( self, logits ):
        """Applies the skew function from Huang et al. and a mask to derive the lower triangular
        portion of the relative positional logits.

        This is pulled directly from the paper Music Transformer: Generating Music with Long-Term
        Structure by Huang et al.
        """
        T = logits.size( dim=-2 )

        # Pad a column to the left.
        logits = F.pad( logits, (1, 0), mode="constant", value=0 )

        # Reshape to (T+1) x T.
        logits = logits.reshape( -1, T + 1, T )

        # Slice off top row and return the lower triangular portion.
        return logits[:, 1:, :].tril()

    def compute_relative_logits_upper( self, queries ):
        """Computes the lower triangular portion of the relative positional logits."""
        T = queries.size( dim=-2 )

        # NOTE: Recall that rpe_pos does NOT contain an embedding for relative distance of zero
        # because that is already stored and learned in rpe_neg. However, (R_u)^T must have a column
        # representing that embedding (namely the first column).
        #
        # Since any value derived from the first column of (R_u)^T is ultimately masked out anyway,
        # it doesn't matter what we place there. For simplicity, the result of executing either
        # branch of this conditional is R_u_trans will be initialized to a tensor containing the
        # last T-1 columns of (R_u)^T. After the conditional, we pad R_u_trans on the left with one
        # column of zeros to finish building the complete (R_u)^T.
        #
        if T <= (self.max_relative_distance + 1):
            # Crop out the positive relative position embeddings we need to build the last T-1
            # columns of (R_u)^T.
            R_u_trans = self.rpe_pos[:, :, :(T - 1)]
        else:
            # Need to pad out the positive relative position embeddings, replicating the embedding
            # for distance C across the last T-C-1 columns of (R_u)^T.
            pad_amount = T - self.max_relative_distance - 1
            R_u_trans = F.pad( self.rpe_pos, (0, pad_amount), mode="replicate" )

        # Add in a dummy column for relative distance of zero.
        R_u_trans = F.pad( R_u_trans, (1, 0), mode="constant", value=0 )
        return self.skew_upper( torch.bmm( queries, R_u_trans ) )

    def skew_upper( self, logits ):
        """Applies the skew function based on the one from Huang et al. and a mask to derive the
        upper triangular portion of the relative positional logits above the main diagonal.

        This is inspired by the skew function described in the paper Music Transformer: Generating
        Music with Long-Term Structure by Huang et al.
        """
        T = logits.size( dim=-2 )

        # Pad a column to the right.
        logits = F.pad( logits, (0, 1), mode="constant", value=0 )

        # Reshape to (T+1) x T.
        logits = logits.reshape( -1, T + 1, T )

        # Slice off the bottom row and return the upper triangular portion above the main diagonal.
        return logits[:, :-1, :].triu( diagonal=1 )
