import bz2
import pathlib
import pickle
import random
import time

import torch
import torch.nn.functional as F

from ..hyperp import Hyperparameters
from .loss import LabelSmoothing
from ..model import MusicTransformer, create_attention_mask
from .optimizer import StandardOptimizer

def dump_compressed_pickle( obj, output_path ):
    with bz2.open( output_path, "wb" ) as output_file:
        output_file.write( pickle.dumps( obj ) )

def load_compressed_pickle( input_path ):
    with bz2.open( input_path, "rb" ) as input_file:
        return pickle.loads( input_file.read() )

def checkpoint_model( model, checkpoint_path ):
    # Move any existing checkpoint to a backup file because don't trust computers.
    checkpoint_path = pathlib.Path( checkpoint_path )
    if checkpoint_path.exists():
        checkpoint_path.replace( checkpoint_path.with_suffix( ".bak" ) )
    torch.save( model.state_dict(), str( checkpoint_path ) )

def train_model( data_path, model, loss_criterion, optimizer, checkpoint_path,
    number_epochs=1, checkpoint_interval_sec=600 ):
    data_sets = load_compressed_pickle( data_path )

    # We'll randomize the order of the training data.
    training_indices = list( range( len( data_sets["train"] ) ) )
    random.shuffle( training_indices )

    # Enter training mode to enable certain layers like dropouts.
    model.train()
    total_steps = 0

    for epoch_idx in range( number_epochs ):
        start_time = time.monotonic()
        epoch_loss = 0.0
        epoch_steps = 0

        for training_idx in training_indices:
            data_sequence = data_sets["train"][training_idx]
            source_sequence = data_sequence[:-1]
            target_sequence = data_sequence[1:]
            source_length = source_sequence.size( 0 )

            # Move to the GPU if available.
            if torch.cuda.is_available():
                source_sequence = source_sequence.cuda()
                target_sequence = target_sequence.cuda()

            # Create a new decoder self-attention mask for this decode step. Move it to the GPU
            # if available.
            attention_mask = create_attention_mask( source_length, source_length )
            if torch.cuda.is_available():
                attention_mask = attention_mask.cuda()

            # Run one step of the model.
            model_output = model( source_sequence=source_sequence, source_mask=attention_mask, encode_only=True )
            loss = loss_criterion( model_output, target_sequence )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1
            total_steps += 1

            # Output every 100 steps just to show we're making progress.
            if epoch_steps % 100 == 0:
                print( (f"Processed {epoch_steps} steps on epoch {epoch_idx + 1}. "
                    f"Current average epoch loss: {(epoch_loss / epoch_steps):.5}.") )

            # Checkpoint and report current status if we've hit our checkpoint interval.
            current_time = time.monotonic()
            elapsed_time = current_time - start_time
            if elapsed_time >= checkpoint_interval_sec:
                print( (f"Checkpointing after {epoch_steps} steps on epoch {epoch_idx + 1}. "
                    f"Current epoch loss for epoch: {epoch_loss:.5}. "
                    f"Current average epoch loss: {(epoch_loss / epoch_steps):.5}. "
                    f"Most recent loss: {loss.item():.5}.") )
                checkpoint_model( model, checkpoint_path )
                start_time = time.monotonic()

        print( (f"Completed epoch {epoch_idx + 1} with total epoch loss of {epoch_loss:.5} "
            f"and average epoch loss of {(epoch_loss / epoch_steps):.4}. Checkpointing.") )
        checkpoint_model( model, checkpoint_path )

    print( f"Training complete after {total_steps} steps." )

def test_model( data_path, model, loss_criterion ):
    """Computes the test data loss of the given model.

    The test data stored in the given data path is run through the model. It output is compared
    against the test data's target output, and the loss is computed.

    Args:
        data_path: Path to the .pkl.bz2 data file containing the test data.
        model: Music Transformer model to test.
        loss_criterion: Function of the model output and expected output that will be used to
            compute the test loss of the given model.

    Returns:
        A 2-element tuple containing the following items:
            * Total accumulated loss of the model.
            * Number of steps over which the total loss was accumulated. This can be used to compute
              an average loss metric.
    """
    data_sets = load_compressed_pickle( data_path )

    # Enter evaluation mode to disable certain layers like dropouts.
    model.eval()
    total_steps = 0
    total_loss = 0.0

    with torch.no_grad():
        for test_data in data_sets["test"]:
            # For testing purposes, we don't use teacher forcing but instead feed the actual output
            # of the decoder back into its input. We need to initialize our actual target sequence
            # (the one we actually feed into the decoder) to be the same size as our expected one
            # and initialize its first element to the start token.
            source_sequence = test_data["source_sequence"]
            expected_target_sequence = test_data["target_sequence"]

            actual_target_sequence = torch.empty_like( expected_target_sequence )
            actual_target_sequence[0] = expected_target_sequence[0]

            # Move to the GPU if available.
            if torch.cuda.is_available():
                source_sequence = source_sequence.cuda()
                expected_target_sequence = expected_target_sequence.cuda()
                actual_target_sequence = actual_target_sequence.cuda()

            # Run the encoder once for all decode steps we'll perform over the current test data set.
            model( input_sequence=source_sequence )

            # Run the decoder over all tokens in the target sequence up to by not including the stop
            # token, which should be the last token in the sequence.
            for target_idx in range( expected_target_sequence.size( -1 ) - 1 ):
                next_token_idx = target_idx + 1
                current_expected_target_sequence = expected_target_sequence[:next_token_idx]
                current_actual_target_sequence = actual_target_sequence[:next_token_idx]
                current_target_length = current_expected_target_sequence.size( 0 )

                # Create a new decoder self-attention mask for this decode step. Move it to the GPU
                # if available.
                attention_mask = create_attention_mask( current_target_length, current_target_length )
                if torch.cuda.is_available():
                    attention_mask = attention_mask.cuda()

                # Run one step of the decoder.
                model_output = model( output_sequence=current_actual_target_sequence, attention_mask=attention_mask )

                # Compute the loss. Only look at the last row of scores since that corresponds to
                # the newest output token predicted.
                #
                # NOTE: Specify -1: slice as first dimension, otherwise pytorch will collapse the
                # result from a 1xN to just an N-element vector, which screws up the loss function.
                # Also the target needs to be a vector instead of a scalar since pytorch loss
                # functions typically expect them to correspond with the output of mini-batches, but
                # we currently don't do any batching with our Transformer model.
                #
                next_item_scores = model_output[-1:, :]
                loss = loss_criterion( next_item_scores, expected_target_sequence[next_token_idx].view( 1 ) )
                total_loss += loss.item()
                total_steps += 1

                # Calculate the actual token predicted by the decoder and add it to the actual
                # target sequence decoded by the decoder.
                actual_target_sequence[next_token_idx] = next_item_scores.argmax().item()

                # Output every 20 steps just to show we're making progress.
                if total_steps % 20 == 0:
                    print( (f"Processed {total_steps} steps. "
                        f"Current average loss: {(total_loss / total_steps):.5}. "
                        f"Current total loss: {total_loss:.5}.") )

    print( f"Training complete after {total_steps} steps." )
    return (total_loss, total_steps)

def run_standard_trainer( data_path, checkpoint_path, vocab_size, weights_path=None,
    number_epochs=1, checkpoint_interval_sec=600, hyper=None ):
    """Runs the standard Music Transformer trainer.

    Args:
        data_path: Path to the .pkl.bz2 data file containing the training data.
        checkpoint_path: Path to which model weights will be periodically checkpointed.
        vocab_size: The size of the vocabulary used by the training data.
        weights_path: Optional path to a file containing model weights previously checkpointed by
            this trainer. This allows training to resume from a previous checkpoint.
        number_epochs: Number of times to loop over the training set.
        checkpoint_interval_sec: Number of seconds to wait before checkpointing the model weights.
    """
    hyper = hyper or Hyperparameters( vocab_size )
    model = MusicTransformer( hyper )

    # If a weights path is given, load the pre-trained weights into our model.
    if weights_path:
        model.load_state_dict( torch.load( weights_path, map_location="cpu" ) )

    # Run on the GPU if it's available.
    if torch.cuda.is_available():
        print( "Using GPU for training" )
        model.cuda()

    optimizer = StandardOptimizer( model.parameters(), hyper.embedding_size )
    #loss_criterion = F.cross_entropy
    loss_criterion = LabelSmoothing( vocab_size )
    train_model( data_path, model, loss_criterion, optimizer, checkpoint_path, number_epochs, checkpoint_interval_sec )

    # Ensure the trained model parameters are back on the CPU before checkpointing.
    model.cpu()
    checkpoint_model( model, checkpoint_path )

def run_standard_tester( data_path, weights_path, vocab_size ):
    """Runs the standard tester on a trained Music Transformer.

    This will run a Music Transformer whose trained weights are given in the weights path against
    the test data set stored at the given data path. Then the model's output is compared against
    the expected output, and the total model loss over the test data set is computed and reported.

    Args:
        data_path: Path to the .pkl.bz2 data file containing the test data.
        weights_path: Path to a file containing the model weights of the trained Music Transformer.
        vocab_size: Size of the vocabulary used by the test data.

    Returns:
        A 2-element tuple containing the following items:
            * Total accumulated loss of the model.
            * Number of steps over which the total loss was accumulated. This can be used to compute
              an average loss metric.
    """
    hyper = Hyperparameters( vocab_size )
    model = MusicTransformer( hyper )
    model.load_state_dict( torch.load( weights_path, map_location="cpu" ) )

    # Run on the GPU if it's available.
    if torch.cuda.is_available():
        print( "Using GPU for testing." )
        model.cuda()

    #loss_criterion = F.cross_entropy
    loss_criterion = LabelSmoothing( vocab_size )
    return test_model( data_path, model, loss_criterion )
