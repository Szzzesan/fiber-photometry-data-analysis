import datetime

"""
[head, trial, cue, reward, lick, leave, start, port1, port2] = get_bools(events)
"""


def get_bools(events):
    head = events.key == 'head'
    trial = events.key == 'trial'
    cue = events.key == 'LED'
    reward = events.key == 'reward'
    lick = events.key == 'lick'
    off = events.value == 0
    on = events.value == 1
    port1 = events.port == 1
    port2 = events.port == 2
    valid_head = events.is_valid
    return [head, trial, cue, reward, lick, off, on, port1, port2, valid_head]


def read_datetime(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d_%H-%M-%S')


def write_datetime(date_time):
    return date_time.strftime('%Y-%m-%d_%H-%M-%S')
