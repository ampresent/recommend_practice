import gzip
import pickle
import sys

def load_daily_events(day):
  db_dir = '.'
  data_file = '{}/watch_day{}.pkl.gz'.format(db_dir, day)
  events = pickle.load(gzip.open(data_file, 'r'))

  return events


if __name__ == '__main__':
    d = int(sys.argv[1])
    events = load_daily_events(d)
    print '[D] loaded num of events = {}'.format(len(events))
    print events

