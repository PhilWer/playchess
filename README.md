# The ROS-chessbot

This package implements a robotic assistant for motion-impaired chess player. To date, the robot cannot play autonomously. The main goal of the application is the robust execution of the moves requested by the player via an accessible interface (joystick, touchpad, mouse, vocal command) using the standard chess equipment used in official competitions.

# How to install

- Install the required libraries from `requirements.txt`... TODO.

- Adapt the path in the scripts in `scripts` folder, to make `PLAYCHESS_PKG_DIR` point to the actual location of the package in the local machine. TODO. Set automatically at launch.

- Adapt the shebangs to point to a valid Python interpreter

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

## Docs

- [ ] add documentation
  - [ ] add documentation inside the code
  - [x] add user instructions to the README

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
- ~~[ ] remove Open3D dependency~~

## Usability

- [ ] add vocal control

## Mainteinance

- [ ] setup TIAGo simulation(s)
  - [ ] complete game setup (chessboard with pieces, box, and clock)
  - [ ] initial setup (complete game setup with ArUCo)
  - [ ] chessboard only
  - [ ] chess-set only
- [ ] tidy the package (e.g. temporary files to `tmp` -or similar- folder)
  - [ ] `PointcloudForCalibration` should move to `tmp`
  - [ ] `gui\images\markers_localization.png` should move to `tmp`
  - [ ] `gui\images\segmentation_example.png` should move to `tmp` and renamed to `square_centers.png`
  - [ ] nest scripts (e.g. `gui`, `vision`, `motion`, `interface`, `config`)
- [ ] create a map of GUI screens and associated script in `scripts/gui`
