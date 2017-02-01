# tilty
A game that is controlled by tilting your head. Built in Python using OpenCV and Pygame.

The file eye_tracker.py has class EyeTracker that is meant to be the core of the concept. Presently, since detection is slow, we use it as sparingly as possible, maybe once a second or so (yet to be tweaked), and use camshift and meanshift to track eye positions in the remaining frames.

Still in testing phase. Once the eye tracking is stable enough, we will move to integrating it with Pygame and build out the game. 
