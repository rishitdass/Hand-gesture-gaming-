# Hand Gesture Game Control

This project implements a hand gesture recognition system for controlling games using MediaPipe and OpenCV. The goal is to detect specific hand gestures and map them to corresponding keyboard actions (W, A, S, D) for game control.

## Features
- Real-time hand gesture detection using MediaPipe.
- Custom gesture matching algorithm to detect W, A, S, D movements.
- Control games by mapping gestures to keyboard key presses using `control` module.
- Multi-hand support to differentiate between gestures performed by the left and right hand.

## Usage

### Running the Application

To launch the application, run the script using your webcam. This will start detecting gestures and control the game based on predefined mappings. The system uses MediaPipe to detect hand landmarks and compares them to predefined gestures (A, W, S, D, STOP).

### Game Integration

This gesture control system can be used to control games or other applications that respond to keyboard inputs. You can modify the key mapping to control various functions in your game.

### Control Mapping

The gestures correspond to the following actions:

- **Left Hand Gesture**: Matches with `A` (Move Left) and  `D` (Move Right)
- **Right Hand Gesture**: Matches with `W` (Move Forward) and  `A` (Move backward)
- **Both Hands Gesture**: Matches with `STOP` to stop movement

## Code Structure

- **hand_gesture_control.py**: The main script that captures the video feed, detects hand gestures, and controls keyboard inputs based on predefined gesture mappings.
- **control.py**: Contains functions to simulate key presses.

## Code Overview

### Gesture Detection

The script uses MediaPipe to detect hand landmarks in the video feed. The landmarks are compared to predefined reference gestures using a threshold to ensure accuracy.

### Predefined Gestures
Custom gestures are predefined by specific hand landmark coordinates, which are compared to detected hand landmarks in the video feed.

### Gesture Matching
Detected hand landmarks are compared against the predefined gestures. If a match is found, the corresponding key (W, A, S, D) is pressed. If no match is found, the key is released.

### Key Mapping
Keys are mapped using a hexadecimal keycode system. The mappings for W, A, S, D keys are defined in the code using the control library.

### Real-time Webcam Feed
OpenCV is used to capture the video stream from the webcam. The hand landmarks are processed and overlaid on the live video feed, along with the FPS (frames per second).


## Requirements
To run this project, the following libraries are required:

- Python 3.x
- OpenCV
- MediaPipe
- Numpy
- A library to simulate keyboard presses (e.g., pyinput or control.py in this case)

## Customization
- Adding new gestures: Modify the reference_gesture_X dictionaries in the code to define new gestures.
- Gesture matching sensitivity: Adjust the threshold parameter in the is_gesture_match function to fine-tune how closely detected gestures must match the predefined ones.
- Key mappings: Change the id_hex dictionary to map different gestures to other keys.
