# The ROS-chessbot

This package implements a robotic assistant for motion-impaired chess player. To date, the robot cannot play autonomously. The main goal of the application is the robust execution of the moves requested by the player via an accessible interface (joystick, touchpad, mouse, vocal command) using the standard chess equipment used in official competitions.

# How to use

## TIAGo robot

See the [instructions](docs/TIAGo_instructions.md) to run the application on the PAL Robotics TIAGo robot.

## Other platforms

_To do._

# References

_Add any._

# Cite this work

If our work gave you useful insight on how to implement your chess-bot, please cite the related article
_Add article in plain text._
_Add article in `bib` format._

# TODO

## Vaishakh

- [ ] add a method to the vision class the runs the whole computer-vision pipeline. The output should be formatted as follows:
  - centers (or corners) of the squares (expressed in pixels), organized as in the chessboard (i.e. in a 8x8 matrix).
  - the image of the scene with the centers (or corners) of the squares overlapped.
  - the image of the scene with the chessboard corners of the chessboard overlapped.
  - the number of detected squares (in your case it should be always 64). We will remove this argument, but now keep it for compatibility.
  - the location fo the chessboard corners.

## Docs

- [ ] add documentation
  - [ ] add documentation inside the code
  - [ ] add user instructions to the README

## Setup

- [ ] remove the need for the manual positioning of the arm during the setup
- [ ] improve move recognition/game tracking
  - [ ] reduce the setup steps
  - [ ] make the tracking more robust
- [ ] improve game saving/new game
  - [ ] remove the need to duplicate the game situation file
  - [ ] remove the need to manually edit the `ready` flag
  - [ ] unify `tiago_chess.launch` and `tiago_chess_pronto.launch`
  - [ ] implement a GUI screen with the "New game"/"Load game"

## Code

- [ ] refactor code
  - [ ] avoid duplicates
  - [ ] expose topics for remapping
  - [ ] remove hardcodings
- [ ] remove Open3D dependency

## Usability

- [ ] add vocal control

## Mainteinance

- [ ] setup TIAGo simulation(s)
  - [ ] complete game setup (chessboard with pieces, box, and clock)
  - [ ] initial setup (complete game setup with ArUCo)
  - [ ] chessboard only
  - [ ] chess-set only
- [ ] tidy the package (e.g. temporary files to `temp` -or similar- folder)
