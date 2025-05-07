lines = open('../config.txt').read().splitlines()
print("Using model {}".format(lines[2]))
ENGINE_PATH = f"../models/{lines[2]}"
MEASUREMENT_ROW = 240  # center row of image (image is 640w, 480h)

MOGO_TEST_FILES = ['../data/ours/feed2_640x480_311.jpeg', '../data/ours/feed2_640x480_295.jpeg', '../data/ours/feed2_640x480_289.jpeg',
                   '../data/ours/feed2_640x480_244.jpeg', '../data/ours/feed2_640x480_232.jpeg', '../data/ours/feed2_640x480_144.jpeg', '../data/ours/feed2_640x480_124.jpeg']
