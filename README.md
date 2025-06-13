# Assistant for the Blind

A real-time environment description system that helps visually impaired people understand their surroundings using computer vision and text-to-speech technology.

## Overview

This system uses a BLIP (Bootstrapping Language-Image Pre-training) model to generate natural language descriptions of the environment captured through a camera. The descriptions are then converted to speech, providing real-time audio feedback to visually impaired users.

## System Architecture

The project consists of two main components:

1. **Sender (Jetson Nano)**

   - Captures video feed from camera
   - Processes frames using BLIP model
   - Generates environment descriptions
   - Sends descriptions to receiver

2. **Receiver (Any Computer)**
   - Receives descriptions from sender
   - Converts text to speech
   - Plays audio feedback to user

## Code Structure

- `sender.py` - Main program running on Jetson Nano for environment capture and description
- `TTS-receiver.py` - Text-to-speech receiver and audio playback
- `setup.sh` - Installation script for required dependencies
- `run.sh` - Script to run the receiver
- `start.sh` - Script to start the sender on Jetson Nano
- `system-install.sh` - System-level dependencies installation
- `generated_audio/` - Directory for temporary audio files

## Requirements

- Jetson Nano (for sender)
- Camera
- Python 3.9+
- Internet connection for initial model download
- Speakers/headphones for audio output

## Quick Start

1. Run `system-install.sh` to install system dependencies
2. Run `setup.sh` to install Python dependencies
3. On Jetson Nano: Run `start.sh` to start the sender
4. On receiver computer: Run `run.sh` to start the TTS receiver

## Note

The system is designed to run the computationally intensive BLIP model on the Jetson Nano (sender) while keeping the text-to-speech functionality on a separate device (receiver) for better performance and flexibility.
