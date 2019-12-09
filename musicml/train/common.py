import pickle
import random
import time

import torch
import torch.nn.functional as F

from ..hyperp import Hyperparameters
from ..model import MusicTransformer, create_attention_mask
from .optimizer import StandardOptimizer

def train_model( data_path, model, loss_criterion, optimizer, checkpoint_path,
    number_epochs=2, checkpoint_interval_sec=120 ):
    with open( data_path, "rb" ) as data_file:
        data_sets = pickle.load( data_file )

    # We'll randomize the order of the training data.
    training_indices = list( range( len( data_sets["train"] ) ) )
    random.shuffle( training_indices )
    total_steps = 0

    for epoch_idx in range( number_epochs ):
        start_time = time.monotonic()
        total_loss = 0.0

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

            # Encode the source sequence.
            #model( input_sequence=source_sequence )

            # Run the decoder over all tokens in the target sequence up to but not including the
            # stop token, which should be the last token in the sequence.
            for target_idx in range( target_sequence.size( -1 ) - 1 ):
                print( f"Step #{total_steps + 1}" )
                next_token_idx = target_idx + 1
                current_target_sequence = target_sequence[:next_token_idx]
                current_target_length = current_target_sequence.size( 0 )

                # Create a new decoder self-attention mask for this decode step. Move it to the GPU
                # if avaiable.
                attention_mask = create_attention_mask( current_target_length, current_target_length )
                if torch.cuda.is_available():
                    attention_mask = attention_mask.cuda()

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
                total_loss += loss.item()
                total_steps += 1

                # Checkpoint and report current status if we've hit our checkpoint interval.
                current_time = time.monotonic()
                elapsed_time = current_time - start_time
                if elapsed_time >= checkpoint_interval_sec:
                    print( (f"Checkpointing after {total_steps} steps on epoch {epoch_idx + 1}. "
                        f"Current total loss for epoch: {total_loss:.4}. "
                        f"Current average loss: {(total_loss / total_steps):.4}. "
                        f"Most recent loss: {loss.item():.4}.") )
                    torch.save( model.state_dict(), checkpoint_path )
                    start_time = current_time

        print( (f"Completed epoch {epoch_idx + 1} with total loss of {total_loss:.4} "
            f"and average loss of {(total_loss / total_steps):.4}. Checkpointing.") )
        torch.save( model.state_dict(), checkpoint_path )

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