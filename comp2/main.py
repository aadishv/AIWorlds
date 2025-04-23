from camera import Worker30
from inference import Model
import threading, time, signal, sys

shutdown = threading.Event()

def handle_sigterm(signum, frame):
    shutdown.set()

signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)

model  = Model("/home/aadish/AIWorlds/comp/utils/best.engine")
worker = Worker30()

threads = []
# inference reader
t1 = threading.Thread(target=model.main1, daemon=True)
threads.append(t1)
# inference webserver
t2 = threading.Thread(target=model.main2, daemon=True)
threads.append(t2)
# camera reader
t3 = threading.Thread(target=worker.main1, daemon=True)
threads.append(t3)
# camera webserver
t4 = threading.Thread(target=worker.main2, daemon=True)
threads.append(t4)

# start them all
for t in threads:
    t.start()

# now block here until CTRL‑C or SIGTERM
shutdown.wait()

# clean up
model.close()
# if your threads check shutdown flag, they can exit cleanly
for t in threads:
    t.join(timeout=1)
