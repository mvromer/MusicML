import csv
import os
import pathlib
import random

from mido import MidiFile, tick2second

Vocabulary = [
    "NOTE_ON<0>",
    "NOTE_ON<1>",
    "NOTE_ON<2>",
    "NOTE_ON<3>",
    "NOTE_ON<4>",
    "NOTE_ON<5>",
    "NOTE_ON<6>",
    "NOTE_ON<7>",
    "NOTE_ON<8>",
    "NOTE_ON<9>",
    "NOTE_ON<10>",
    "NOTE_ON<11>",
    "NOTE_ON<12>",
    "NOTE_ON<13>",
    "NOTE_ON<14>",
    "NOTE_ON<15>",
    "NOTE_ON<16>",
    "NOTE_ON<17>",
    "NOTE_ON<18>",
    "NOTE_ON<19>",
    "NOTE_ON<20>",
    "NOTE_ON<21>",
    "NOTE_ON<22>",
    "NOTE_ON<23>",
    "NOTE_ON<24>",
    "NOTE_ON<25>",
    "NOTE_ON<26>",
    "NOTE_ON<27>",
    "NOTE_ON<28>",
    "NOTE_ON<29>",
    "NOTE_ON<30>",
    "NOTE_ON<31>",
    "NOTE_ON<32>",
    "NOTE_ON<33>",
    "NOTE_ON<34>",
    "NOTE_ON<35>",
    "NOTE_ON<36>",
    "NOTE_ON<37>",
    "NOTE_ON<38>",
    "NOTE_ON<39>",
    "NOTE_ON<40>",
    "NOTE_ON<41>",
    "NOTE_ON<42>",
    "NOTE_ON<43>",
    "NOTE_ON<44>",
    "NOTE_ON<45>",
    "NOTE_ON<46>",
    "NOTE_ON<47>",
    "NOTE_ON<48>",
    "NOTE_ON<49>",
    "NOTE_ON<50>",
    "NOTE_ON<51>",
    "NOTE_ON<52>",
    "NOTE_ON<53>",
    "NOTE_ON<54>",
    "NOTE_ON<55>",
    "NOTE_ON<56>",
    "NOTE_ON<57>",
    "NOTE_ON<58>",
    "NOTE_ON<59>",
    "NOTE_ON<60>",
    "NOTE_ON<61>",
    "NOTE_ON<62>",
    "NOTE_ON<63>",
    "NOTE_ON<64>",
    "NOTE_ON<65>",
    "NOTE_ON<66>",
    "NOTE_ON<67>",
    "NOTE_ON<68>",
    "NOTE_ON<69>",
    "NOTE_ON<70>",
    "NOTE_ON<71>",
    "NOTE_ON<72>",
    "NOTE_ON<73>",
    "NOTE_ON<74>",
    "NOTE_ON<75>",
    "NOTE_ON<76>",
    "NOTE_ON<77>",
    "NOTE_ON<78>",
    "NOTE_ON<79>",
    "NOTE_ON<80>",
    "NOTE_ON<81>",
    "NOTE_ON<82>",
    "NOTE_ON<83>",
    "NOTE_ON<84>",
    "NOTE_ON<85>",
    "NOTE_ON<86>",
    "NOTE_ON<87>",
    "NOTE_ON<88>",
    "NOTE_ON<89>",
    "NOTE_ON<90>",
    "NOTE_ON<91>",
    "NOTE_ON<92>",
    "NOTE_ON<93>",
    "NOTE_ON<94>",
    "NOTE_ON<95>",
    "NOTE_ON<96>",
    "NOTE_ON<97>",
    "NOTE_ON<98>",
    "NOTE_ON<99>",
    "NOTE_ON<100>",
    "NOTE_ON<101>",
    "NOTE_ON<102>",
    "NOTE_ON<103>",
    "NOTE_ON<104>",
    "NOTE_ON<105>",
    "NOTE_ON<106>",
    "NOTE_ON<107>",
    "NOTE_ON<108>",
    "NOTE_ON<109>",
    "NOTE_ON<110>",
    "NOTE_ON<111>",
    "NOTE_ON<112>",
    "NOTE_ON<113>",
    "NOTE_ON<114>",
    "NOTE_ON<115>",
    "NOTE_ON<116>",
    "NOTE_ON<117>",
    "NOTE_ON<118>",
    "NOTE_ON<119>",
    "NOTE_ON<120>",
    "NOTE_ON<121>",
    "NOTE_ON<122>",
    "NOTE_ON<123>",
    "NOTE_ON<124>",
    "NOTE_ON<125>",
    "NOTE_ON<126>",
    "NOTE_ON<127>",
    "NOTE_OFF<0>",
    "NOTE_OFF<1>",
    "NOTE_OFF<2>",
    "NOTE_OFF<3>",
    "NOTE_OFF<4>",
    "NOTE_OFF<5>",
    "NOTE_OFF<6>",
    "NOTE_OFF<7>",
    "NOTE_OFF<8>",
    "NOTE_OFF<9>",
    "NOTE_OFF<10>",
    "NOTE_OFF<11>",
    "NOTE_OFF<12>",
    "NOTE_OFF<13>",
    "NOTE_OFF<14>",
    "NOTE_OFF<15>",
    "NOTE_OFF<16>",
    "NOTE_OFF<17>",
    "NOTE_OFF<18>",
    "NOTE_OFF<19>",
    "NOTE_OFF<20>",
    "NOTE_OFF<21>",
    "NOTE_OFF<22>",
    "NOTE_OFF<23>",
    "NOTE_OFF<24>",
    "NOTE_OFF<25>",
    "NOTE_OFF<26>",
    "NOTE_OFF<27>",
    "NOTE_OFF<28>",
    "NOTE_OFF<29>",
    "NOTE_OFF<30>",
    "NOTE_OFF<31>",
    "NOTE_OFF<32>",
    "NOTE_OFF<33>",
    "NOTE_OFF<34>",
    "NOTE_OFF<35>",
    "NOTE_OFF<36>",
    "NOTE_OFF<37>",
    "NOTE_OFF<38>",
    "NOTE_OFF<39>",
    "NOTE_OFF<40>",
    "NOTE_OFF<41>",
    "NOTE_OFF<42>",
    "NOTE_OFF<43>",
    "NOTE_OFF<44>",
    "NOTE_OFF<45>",
    "NOTE_OFF<46>",
    "NOTE_OFF<47>",
    "NOTE_OFF<48>",
    "NOTE_OFF<49>",
    "NOTE_OFF<50>",
    "NOTE_OFF<51>",
    "NOTE_OFF<52>",
    "NOTE_OFF<53>",
    "NOTE_OFF<54>",
    "NOTE_OFF<55>",
    "NOTE_OFF<56>",
    "NOTE_OFF<57>",
    "NOTE_OFF<58>",
    "NOTE_OFF<59>",
    "NOTE_OFF<60>",
    "NOTE_OFF<61>",
    "NOTE_OFF<62>",
    "NOTE_OFF<63>",
    "NOTE_OFF<64>",
    "NOTE_OFF<65>",
    "NOTE_OFF<66>",
    "NOTE_OFF<67>",
    "NOTE_OFF<68>",
    "NOTE_OFF<69>",
    "NOTE_OFF<70>",
    "NOTE_OFF<71>",
    "NOTE_OFF<72>",
    "NOTE_OFF<73>",
    "NOTE_OFF<74>",
    "NOTE_OFF<75>",
    "NOTE_OFF<76>",
    "NOTE_OFF<77>",
    "NOTE_OFF<78>",
    "NOTE_OFF<79>",
    "NOTE_OFF<80>",
    "NOTE_OFF<81>",
    "NOTE_OFF<82>",
    "NOTE_OFF<83>",
    "NOTE_OFF<84>",
    "NOTE_OFF<85>",
    "NOTE_OFF<86>",
    "NOTE_OFF<87>",
    "NOTE_OFF<88>",
    "NOTE_OFF<89>",
    "NOTE_OFF<90>",
    "NOTE_OFF<91>",
    "NOTE_OFF<92>",
    "NOTE_OFF<93>",
    "NOTE_OFF<94>",
    "NOTE_OFF<95>",
    "NOTE_OFF<96>",
    "NOTE_OFF<97>",
    "NOTE_OFF<98>",
    "NOTE_OFF<99>",
    "NOTE_OFF<100>",
    "NOTE_OFF<101>",
    "NOTE_OFF<102>",
    "NOTE_OFF<103>",
    "NOTE_OFF<104>",
    "NOTE_OFF<105>",
    "NOTE_OFF<106>",
    "NOTE_OFF<107>",
    "NOTE_OFF<108>",
    "NOTE_OFF<109>",
    "NOTE_OFF<110>",
    "NOTE_OFF<111>",
    "NOTE_OFF<112>",
    "NOTE_OFF<113>",
    "NOTE_OFF<114>",
    "NOTE_OFF<115>",
    "NOTE_OFF<116>",
    "NOTE_OFF<117>",
    "NOTE_OFF<118>",
    "NOTE_OFF<119>",
    "NOTE_OFF<120>",
    "NOTE_OFF<121>",
    "NOTE_OFF<122>",
    "NOTE_OFF<123>",
    "NOTE_OFF<124>",
    "NOTE_OFF<125>",
    "NOTE_OFF<126>",
    "NOTE_OFF<127>",
    "TIME_SHIFT<10>",
    "TIME_SHIFT<20>",
    "TIME_SHIFT<30>",
    "TIME_SHIFT<40>",
    "TIME_SHIFT<50>",
    "TIME_SHIFT<60>",
    "TIME_SHIFT<70>",
    "TIME_SHIFT<80>",
    "TIME_SHIFT<90>",
    "TIME_SHIFT<100>",
    "TIME_SHIFT<110>",
    "TIME_SHIFT<120>",
    "TIME_SHIFT<130>",
    "TIME_SHIFT<140>",
    "TIME_SHIFT<150>",
    "TIME_SHIFT<160>",
    "TIME_SHIFT<170>",
    "TIME_SHIFT<180>",
    "TIME_SHIFT<190>",
    "TIME_SHIFT<200>",
    "TIME_SHIFT<210>",
    "TIME_SHIFT<220>",
    "TIME_SHIFT<230>",
    "TIME_SHIFT<240>",
    "TIME_SHIFT<250>",
    "TIME_SHIFT<260>",
    "TIME_SHIFT<270>",
    "TIME_SHIFT<280>",
    "TIME_SHIFT<290>",
    "TIME_SHIFT<300>",
    "TIME_SHIFT<310>",
    "TIME_SHIFT<320>",
    "TIME_SHIFT<330>",
    "TIME_SHIFT<340>",
    "TIME_SHIFT<350>",
    "TIME_SHIFT<360>",
    "TIME_SHIFT<370>",
    "TIME_SHIFT<380>",
    "TIME_SHIFT<390>",
    "TIME_SHIFT<400>",
    "TIME_SHIFT<410>",
    "TIME_SHIFT<420>",
    "TIME_SHIFT<430>",
    "TIME_SHIFT<440>",
    "TIME_SHIFT<450>",
    "TIME_SHIFT<460>",
    "TIME_SHIFT<470>",
    "TIME_SHIFT<480>",
    "TIME_SHIFT<490>",
    "TIME_SHIFT<500>",
    "TIME_SHIFT<510>",
    "TIME_SHIFT<520>",
    "TIME_SHIFT<530>",
    "TIME_SHIFT<540>",
    "TIME_SHIFT<550>",
    "TIME_SHIFT<560>",
    "TIME_SHIFT<570>",
    "TIME_SHIFT<580>",
    "TIME_SHIFT<590>",
    "TIME_SHIFT<600>",
    "TIME_SHIFT<610>",
    "TIME_SHIFT<620>",
    "TIME_SHIFT<630>",
    "TIME_SHIFT<640>",
    "TIME_SHIFT<650>",
    "TIME_SHIFT<660>",
    "TIME_SHIFT<670>",
    "TIME_SHIFT<680>",
    "TIME_SHIFT<690>",
    "TIME_SHIFT<700>",
    "TIME_SHIFT<710>",
    "TIME_SHIFT<720>",
    "TIME_SHIFT<730>",
    "TIME_SHIFT<740>",
    "TIME_SHIFT<750>",
    "TIME_SHIFT<760>",
    "TIME_SHIFT<770>",
    "TIME_SHIFT<780>",
    "TIME_SHIFT<790>",
    "TIME_SHIFT<800>",
    "TIME_SHIFT<810>",
    "TIME_SHIFT<820>",
    "TIME_SHIFT<830>",
    "TIME_SHIFT<840>",
    "TIME_SHIFT<850>",
    "TIME_SHIFT<860>",
    "TIME_SHIFT<870>",
    "TIME_SHIFT<880>",
    "TIME_SHIFT<890>",
    "TIME_SHIFT<900>",
    "TIME_SHIFT<910>",
    "TIME_SHIFT<920>",
    "TIME_SHIFT<930>",
    "TIME_SHIFT<940>",
    "TIME_SHIFT<950>",
    "TIME_SHIFT<960>",
    "TIME_SHIFT<970>",
    "TIME_SHIFT<980>",
    "TIME_SHIFT<990>",
    "TIME_SHIFT<1000>",
    "SET_VELOCITY<1>",
    "SET_VELOCITY<2>",
    "SET_VELOCITY<3>",
    "SET_VELOCITY<4>",
    "SET_VELOCITY<5>",
    "SET_VELOCITY<6>",
    "SET_VELOCITY<7>",
    "SET_VELOCITY<8>",
    "SET_VELOCITY<9>",
    "SET_VELOCITY<10>",
    "SET_VELOCITY<11>",
    "SET_VELOCITY<12>",
    "SET_VELOCITY<13>",
    "SET_VELOCITY<14>",
    "SET_VELOCITY<15>",
    "SET_VELOCITY<16>",
    "SET_VELOCITY<17>",
    "SET_VELOCITY<18>",
    "SET_VELOCITY<19>",
    "SET_VELOCITY<20>",
    "SET_VELOCITY<21>",
    "SET_VELOCITY<22>",
    "SET_VELOCITY<23>",
    "SET_VELOCITY<24>",
    "SET_VELOCITY<25>",
    "SET_VELOCITY<26>",
    "SET_VELOCITY<27>",
    "SET_VELOCITY<28>",
    "SET_VELOCITY<29>",
    "SET_VELOCITY<30>",
    "SET_VELOCITY<31>",
    "SET_VELOCITY<32>",
    "START",
    "STOP"
]

def prepare_data( input_path, output_path ):
    input_path = pathlib.Path( input_path )
    output_path = pathlib.Path( output_path )
    input_file_names = input_path.glob( "**/*.midi" )

    for input_file_name in input_file_names:
        # Create a directory to store the converted file.
        relative_input_file_name = input_file_name.relative_to( input_path )
        output_file_name = output_path / relative_input_file_name.with_suffix( ".out" )
        output_file_name.parent.mkdir( parents=True, exist_ok=True )

        # create a parsed file for each file
        with output_file_name.open("w+") as f:
            print(f"Processing input file: {input_file_name}.")
            mid = MidiFile(str(input_file_name))
            shift_delta = 0

            for msg in mid:
                # get tempo of the music from meta
                if (msg.is_meta and msg.type == "set_tempo"):
                    tempo = msg.tempo

                shift_delta += msg.time * 1000

                if (msg.type == "note_on" and msg.velocity != 0):
                    shift_delta = write_time_shifts(f, shift_delta)
                    f.write(f"SET_VELOCITY<{msg.velocity}>\n")
                    f.write(f"NOTE_ON<{msg.note}>\n")

                if (msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)):
                    shift_delta = write_time_shifts(f, shift_delta)
                    f.write(f"NOTE_OFF<{msg.note}>\n")

def write_time_shifts( output_file, shift_delta, shift_resolution_ms=10 ):
    """Write TIME_SHIFT messages if the given shift_delta exceeds the shift resolution.

    Args:
        output_file: Text file to write the TIME_SHIFT messages to.
        shift_delta: Number of milliseconds that have elapsed in the MIDI event stream since the
            last TIME_SHIFT message.
        shift_resolution_ms: The smallest resolution of elapsed time that can be captured by a
            TIME_SHIFT message.
    """
    # We can encapsulate up to 1 second (1000 ms) of time in each TIME_SHIFT.
    while shift_delta >= 1000:
        output_file.write( "TIME_SHIFT<1000>\n" )
        shift_delta -= 1000

    # Figure out if we can emit another TIME_SHIFT message.
    shift_amount = int(shift_delta // shift_resolution_ms * shift_resolution_ms)
    if shift_amount:
        output_file.write( f"TIME_SHIFT<{shift_amount}>\n" )
        shift_delta -= shift_amount

    return shift_delta

def create_data_sets( input_path, output_path, manifest_path, crop_size=2000, max_files=None ):
    """Creates training and test data sets from the corpus of Piano-e-Competition data located in
    the given input path.

    This uses the canonical training/test split provided by the Maestro v2.0.0 manifest file. The
    input data is converted in the following manner:
        * The file contents are read, and a random crop is selected from the file. The number of
          tokens (lines/events) cropped from the file is equal to the given crop size.
        * The crop is divided in half. The first half is marked as the source sequence, which is the
          sequence fed through the Music Transformer's encoder. The second half is marked as the
          target sequence. It consists of the tokens the decoder is expected to produce.
          * During training, teacher forcing is used such that instead of feeding the decoder's
            output back to its input, the next expected token from the target sequence is fed into
            the decoder during the next decode step.
          * During testing, the actual decoder output is fed back into the decoder during subsequent
            decode steps.
        * The tokens identified for the source and target sequences are converted into formats that
          can be readily consumed by the Music Transformer.
        * Converted training and test data is pickled to the given output path.

    Args:
        input_path: Root folder containing the Piano-e-Competition data previously converted by
            prepare_data.
        output_path: File name under which the training and test data will be saved.
        manifest_path: Path to the Maestro v2.0.0 manifest file designating which files belong to
            the training set versus test set.
        crop_size: Number of tokens to crop from a file when producing source and target sequences.
        max_files: If given, specifies the maximum number of files processed. Otherwise, all
            Piano-e-Competition files are used to produce the training and test data sets.
    """
    data_sets = {
        "train": [],
        "test": []
    }

    input_path = pathlib.Path( input_path )

    with open( manifest_path, newline="" ) as manifest_file:
        manifest_reader = csv.DictReader( manifest_file )
        for row in manifest_reader:
            input_file_name = (input_path / row["midi_filename"]).with_suffix( ".out" )

            with open( input_file_name, "r" ) as input_file:
                input_lines = input_file.readlines()
                number_lines = len( input_lines )

                # Randomly select the offset into this file where we will begin our crop_size crop.
                # If for whatever reason the desired crop size is larger than the input, than just
                # select the entire input.
                if crop_size >= number_lines:
                    source_range = range( 0, number_lines // 2 )
                    target_range = range( source_range.stop, number_lines )
                else:
                    source_length = crop_size // 2
                    target_length = crop_size - source_length
                    source_start = random.randint( 0, number_lines - crop_size )
                    source_range = range( source_start, source_start + source_length )
                    target_range = range( source_range.stop, source_range.stop + target_length )

                for line_idx in source_range:
                    line = input_lines[line_idx]
