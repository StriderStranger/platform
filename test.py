import threading



event_shut = threading.Event()
event_fire = threading.Event()

for i in range(5):
  t = threading.Thread(target=do, args=(event_shut,event_fire), name='thread_{}'.format(i))
  t.start()

if __name__ == "__main__":
  if input("input:") == 'sure':
    event_fire.set()