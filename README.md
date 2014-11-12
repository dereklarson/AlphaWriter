AlphaWriter - Alphanumeric writing with the Leap Motion controller
============

Dependencies:  pygame, numpy/scipy, in addition to a Leap setup.

Simply run `python AlphaWriter.py` with your controller plugged in.
Then, to write, put your index and middle finger together and draw.
When finished writing a character, separate the fingers. To erase,
swipe your hand to the side (keep fingers separated).

Currently, accuracy is poor on lower-case characters (or non-existant) as
well as some trouble cases like {1, l, i, j}. Capital letters do work 
quite well. I generated synthetic data based on system fonts and elastic
distortions, which probably doesn't capture the variations in human
writing as well as it should. Thus, a superior dataset would improve
the performance.
