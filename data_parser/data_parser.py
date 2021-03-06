from mido import MidiFile, tick2second
import os

# get a list of midi files by year
path = './data_parser/maestro-v2.0.0/'
outPath = './data_parser/outputs/'
dataDirDict = {}

for item in os.listdir(path):
    if os.path.isdir(os.path.join(path, item)):
        dataDirDict[item] = []
        for file in os.listdir(os.path.join(path, item)):
            if file.endswith('.midi'):
                dataDirDict[item].append(file)


# create a directory for each year
for year in dataDirDict: 
    currentOutPath = os.path.join(outPath, year)
    os.mkdir(currentOutPath)

    # create a parsed file for each file
    for item in dataDirDict[year]:

        filename = item.replace('.midi', '.out')
        filepath = os.path.join(currentOutPath, filename)
        f = open(filepath,"w+")

        mid = MidiFile(os.path.join(path, year, item))

        ticksPerBeat = mid.ticks_per_beat
        tempo = ''

        for i, track in enumerate(mid.tracks):
            for msg in track:
                # get tempo of the music from meta
                if (msg.is_meta and msg.type == "set_tempo"):
                    tempo = msg.tempo
                
                if (msg.type == "note_on" and msg.velocity != 0):
                    f.write(f"note_on {msg.note}\n")
                    f.write(f"velocity {msg.velocity}\n")
                    time = tick2second(msg.time, ticksPerBeat, tempo)
                    timeInMs = int(time * 1000)
                    f.write(f"time {timeInMs}\n")
                
                if ( msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)):
                    f.write(f"note_off {msg.note}\n")
                    time = tick2second(msg.time, ticksPerBeat, tempo)
                    timeInMs = int(time * 1000)
                    f.write(f"time {timeInMs}\n")

        f.close()