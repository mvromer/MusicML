import bz2
import pathlib
import pickle
import random
import time

import torch
import torch.nn.functional as F

from ..hyperp import Hyperparameters
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
    number_epochs=5, checkpoint_interval_sec=30 ):
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
            training_data = data_sets["train"][training_idx]
            source_sequence = training_data["source_sequence"]
            target_sequence = training_data["target_sequence"]
            target_output = training_data["target_output"]
            source_length = source_sequence.size( 0 )

            # Move to the GPU if available.
            if torch.cuda.is_available():
                source_sequence = source_sequence.cuda()
                target_sequence = target_sequence.cuda()
                target_output = target_output.cuda()

            # Run the decoder over all tokens in the target sequence up to but not including the
            # stop token, which should be the last token in the sequence.
            for target_idx in range( target_sequence.size( -1 ) - 1 ):
                next_token_idx = target_idx + 1
                current_target_sequence = target_sequence[:next_token_idx]
                current_target_length = current_target_sequence.size( 0 )

                # Create a new decoder self-attention mask for this decode step. Move it to the GPU
                # if avaiable.
                attention_mask = create_attention_mask( current_target_length, current_target_length )
                if torch.cuda.is_available():
                    attention_mask = attention_mask.cuda()

                # Encode the source sequence.
                #
                # NOTE: Normally we wouldn't rerun the encoder multiple times like this, but in
                # order for Pytorch to have gradient information on the encoder portion of our
                # model, we need to make sure the encoder is computed again after we do our backward
                # propagation during loss calculation.
                #
                model( input_sequence=source_sequence )

                # Run one step of the decoder.
                model_output = model( output_sequence=current_target_sequence, attention_mask=attention_mask )

                # Compute loss and update parameters. Only look at the last row of scores since that
                # corresponds to the newest output token predicted.
                #
                # NOTE: Specify -1: slice as first dimension, otherwise pytorch will collapse the
                # result from a 1xN to just an N-element vector, which screws up the loss function.
                # Also the target needs to be a vector instead of a scalar since pytorch loss
                # functions typically expect them to correspond with the output of mini-batches, but
                # we currently don't do any batching with our Transformer model.
                #
                loss = loss_criterion( model_output[-1:, :], target_sequence[next_token_idx].view( 1 ) )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1

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

def run_standard_trainer( data_path, checkpoint_path, vocab_size ):
    hyper = Hyperparameters( vocab_size )
    model = MusicTransformer( hyper )

    # Run on the GPU if it's available.
    if torch.cuda.is_available():
        model.cuda()

    optimizer = StandardOptimizer( model.parameters(), hyper.embedding_size )
    loss_criterion = F.cross_entropy
    train_model( data_path, model, loss_criterion, optimizer, checkpoint_path )

    # Ensure the trained model parameters are back on the CPU before checkpointing.
    model.cpu()
    checkpoint_model( model, checkpoint_path )
