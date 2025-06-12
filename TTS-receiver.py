#!/usr/bin/env python3
"""
udp_tts_logger.py

A cross-platform UDP listener that:
1. Waits for incoming UTF-8 string messages over UDP.
2. Appends each message (with a timestamp) to 'messages.log'.
3. Speaks each message aloud using pyttsx3.

Supported on:
    - Windows (uses SAPI5 by default)
    - macOS   (uses NSSpeechSynthesizer by default)
    - Linux   (uses eSpeak or other backends—see below)

Dependencies:
    pip install pyttsx3

On Linux, you may need to install a TTS engine such as:
    sudo apt-get install espeak libespeak1
(or your distro’s equivalent package)

Usage:
    python udp_tts_logger.py
"""

import socket
import pyttsx3
import datetime
import sys
import os
import platform
import json

# Configuration
UDP_IP = "0.0.0.0"       # Listen on all interfaces
UDP_PORT = 5005          # Port to listen on
LOG_FILE = "messages.log"


def init_tts_engine():
    """
    Initialize the pyttsx3 TTS engine in a cross-platform way.
    """
    try:
        engine = pyttsx3.init()
    except Exception as e:
        sys.stderr.write(f"[Error] Cannot initialize TTS engine: {e}\n")
        sys.exit(1)

    # Optional: adjust default voice/rate/volume here if desired
    # rate = engine.getProperty('rate')
    # engine.setProperty('rate', rate - 25)

    return engine


def log_message(message: str):
    """
    Append the given message with a timestamp to the log file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        sys.stderr.write(f"[Error] Failed to write to '{LOG_FILE}': {e}\n")


def main():
    # Print platform info for clarity
    sys.stdout.write(f"Starting on {platform.system()} {platform.release()}\n")

    # Ensure log directory exists (if LOG_FILE is in a subfolder)
    log_dir = os.path.dirname(os.path.abspath(LOG_FILE))
    if log_dir and not os.path.isdir(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            sys.stderr.write(f"[Error] Could not create log directory '{log_dir}': {e}\n")
            sys.exit(1)

    # Initialize TTS engine
    tts_engine = init_tts_engine()

    # Set up UDP socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
    except Exception as e:
        sys.stderr.write(f"[Error] Could not bind UDP socket on {UDP_IP}:{UDP_PORT}: {e}\n")
        sys.exit(1)

    sys.stdout.write(f"Listening for UDP messages on {UDP_IP}:{UDP_PORT}...\n")

    try:
        while True:
            data, addr = sock.recvfrom(4096)  # 4KB buffer
            try:
                message = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                sys.stderr.write(f"[Warning] Received non-UTF8 data from {addr}, skipping.\n")
                continue

            sys.stdout.write(f"Received from {addr}: {message}\n")

            # 1. Log message
            log_message(message)

            # 2. Speak the message
            try:
                try:
                    msg_obj = json.loads(message)
                    text_to_say = msg_obj.get("text", "")
                except Exception as e:
                    sys.stderr.write(f"[Warning] Could not parse JSON or missing 'text': {e}\n")
                    text_to_say = ""
                if text_to_say:
                    tts_engine.say(text_to_say)
                tts_engine.runAndWait()
            except Exception as e:
                sys.stderr.write(f"[Error] TTS engine failed: {e}\n")

    except KeyboardInterrupt:
        sys.stdout.write("\nShutting down (KeyboardInterrupt).\n")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
