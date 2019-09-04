# -*- coding: utf-8 -*-
import sys
import mido
import json
import argparse
from collections import defaultdict

# 480: quater rest/note
# 1920: whole rest/note

def get_length(length):
    length = int(length)
    if length <= 40:
        return 0
    elif length <= 80:
        return 60
    elif length <= 180:
        return 120
    elif length <= 300:
        return 240
    elif length <= 420:
        return 360
    elif length <= 600:
        return 480
    elif length <= 840:
        return 720
    elif length <= 1080:
        return 960
    elif length <= 1320:
        return 1200
    elif length <= 1560:
        return 1440
    elif length <= 1800:
        return 1680
    elif length <= 2880:
        return 1920
    elif length <= 4800:
        return 3840
    elif length <= 4560:
        return 5760
    else:
        return 7680

def convert(mid_file):
    mid = mido.MidiFile(mid_file, ticks_per_beat=480)
    on_queue = defaultdict(list)
    current_position = 0
    notes = []
    for i, track in enumerate(mid.tracks):
        if i == 1:
            for msg in track:
                if msg.type == "note_on":
                    note = msg.note
                    length = get_length(msg.time)
                    current_position += length
                    on_queue[note].append(current_position)


                elif msg.type == "note_off":
                    note = msg.note
                    length = get_length(msg.time)
                    current_position += length

                    if len(on_queue[note]) > 0:
                        start_position = on_queue[note][0]
                        duration = current_position - start_position
                        notes.append((note, start_position, duration))
                        on_queue[note].pop(0)

    fsp = notes[0][1]
    prev_end_position = 0
    notes4generation = [("rest", "7680")]
    for note in notes:
        note_number = note[0]
        start_position = note[1] - fsp
        duration = note[2]
        if start_position < prev_end_position:
            continue
        elif start_position == prev_end_position:
            _temp_duraton = get_length(duration)
            if _temp_duraton > 0:
                notes4generation.append((str(note_number), str(_temp_duraton)))
        else:
            rest_duration = get_length(start_position - prev_end_position)
            rest_start_position = prev_end_position
            notes4generation.append(("rest", str(rest_duration)))
            _temp_duraton = get_length(duration)
            if _temp_duraton > 0:
                notes4generation.append((str(note_number), str(_temp_duraton)))
        prev_end_position = start_position + duration
    notes4generation.append(("rest", "7680"))

    return notes4generation



def main(args):
    notes = convert(args.midi)
    print(json.dumps({"melody":notes}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-midi", "--midi", dest="midi", default="../sample_data/メルト.mid", type=str, help="MIDI file")
    args = parser.parse_args()
    main(args)
