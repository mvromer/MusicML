import pickle
import random

def train_model( data_path, model, loss_criterion, optimizer, checkpoint_path,
    number_epochs=2, checkpoint_interval_sec=120 ):
    with open( data_path, "rb" ) as data_file:
        data_sets = pickle.load( data_file )

    # We'll randomize the order of the training data.
    training_indices = random.shuffle( data_sets["train"] )

    start_time = time.time()

    for epoch_idx in range( number_epochs ):
        for training_idx in training_indices:
            training_data = data_sets["train"][training_idx]
            source_sequence = training_data["source_sequence"]
            target_sequence = training_data["target_sequence"]
            target_output = training_data["target_output"]

            model( input_sequence=source_sequence )

            for target_idx in range( target_sequence.size( -1 ) ):
                pass
