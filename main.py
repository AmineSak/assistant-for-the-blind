#!/usr/bin/env python3
"""
Real-time Environment Description System for Jetson Nano
Captures camera feed, generates text descriptions, and speaks them aloud
"""

import argparse
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pygame
import torch
from transformers import AutoFeatureExtractor, AutoModelForVision2Seq, AutoTokenizer
from TTS.api import TTS


class EnvironmentDescriptionSystem:
    def __init__(self, args):
        self.args = args
        self.running = False
        self.frame_queue = queue.Queue(
            maxsize=2
        )  # Limit queue size to prevent memory issues
        self.text_queue = queue.Queue(maxsize=5)
        self.last_description = ""
        self.last_description_time = 0
        self.description_cooldown = args.cooldown
        self.setup_camera()
        self.setup_vision_model()
        self.setup_tts()
        pygame.mixer.init()

    def setup_camera(self):
        """Initialize the camera with specified resolution"""
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(self.args.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        print(f"Camera initialized at {self.args.width}x{self.args.height}")

    def setup_vision_model(self):
        """Load the image-to-text model"""
        print(f"Loading vision model: {self.args.vision_model}...")

        # For GPU optimization on Jetson
        torch.backends.cudnn.benchmark = True

        # Load model components
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.args.vision_model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vision_model)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.args.vision_model,
            torch_dtype=torch.float16 if self.args.half_precision else torch.float32,
        )

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimize model for inference
        self.model.eval()
        if self.args.half_precision and self.device.type == "cuda":
            self.model = self.model.half()

        print(f"Vision model loaded on {self.device}")

    def setup_tts(self):
        """Initialize text-to-speech engine"""
        print("Setting up TTS...")

        if self.args.tts_engine == "coqui":
            # Coqui TTS - Better quality but more resource-intensive
            self.tts = TTS("tts_models/en/vctk/vits", gpu=torch.cuda.is_available())
        else:
            # We'll use pyttsx3 as fallback (handled in speak_description)
            self.tts = None

        print(f"TTS engine initialized: {self.args.tts_engine}")

    def capture_frames(self):
        """Continuously capture frames from camera"""
        print("Starting frame capture...")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue

            try:
                # Only add new frame if queue isn't full (discard if full)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Skip this frame if queue is full
                pass

        print("Frame capture stopped")

    def process_frames(self):
        """Process frames with vision model to generate descriptions"""
        print("Starting frame processing...")
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)

                # Check if it's time for a new description
                current_time = time.time()
                if (
                    current_time - self.last_description_time
                    < self.description_cooldown
                ):
                    continue

                # Process the frame
                with torch.no_grad():
                    # Prepare image for model
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inputs = self.feature_extractor(
                        images=image, return_tensors="pt"
                    ).to(self.device)

                    # Generate description
                    if self.args.half_precision and self.device.type == "cuda":
                        inputs = {
                            k: v.half() if v.dtype == torch.float32 else v
                            for k, v in inputs.items()
                        }

                    outputs = self.model.generate(
                        **inputs, max_length=50, num_beams=self.args.beam_size
                    )

                    description = self.tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )

                # Only use new descriptions that are different enough
                if self._is_description_different(description):
                    self.last_description = description
                    self.last_description_time = current_time
                    print(f"New description: {description}")

                    # Add to speech queue
                    if not self.text_queue.full():
                        self.text_queue.put(description)

            except queue.Empty:
                # No frames available
                pass
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)

        print("Frame processing stopped")

    def _is_description_different(self, new_desc):
        """Check if new description is significantly different from last one"""
        # Simple implementation - could be improved with semantic comparison
        if not self.last_description:
            return True

        # Check if descriptions are sufficiently different
        # This basic approach can be extended with more sophisticated text similarity metrics
        common_words = set(new_desc.lower().split()) & set(
            self.last_description.lower().split()
        )
        total_words = set(new_desc.lower().split()) | set(
            self.last_description.lower().split()
        )

        if not total_words:
            return False

        difference_ratio = 1 - (len(common_words) / len(total_words))
        return difference_ratio > self.args.similarity_threshold

    def speak_descriptions(self):
        """Convert text descriptions to speech"""
        print("Starting speech synthesis...")
        output_dir = os.path.join(os.getcwd(), "generated_audio")
        os.makedirs(output_dir, exist_ok=True)

        while self.running:
            try:
                text = self.text_queue.get(timeout=1.0)

                # Skip empty descriptions
                if not text.strip():
                    continue

                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = os.path.join(output_dir, f"description_{timestamp}.wav")

                # Generate speech
                if self.args.tts_engine == "coqui" and self.tts:
                    # Use Coqui TTS
                    self.tts.tts_to_file(text, file_path=audio_file)
                else:
                    # Fallback to pyttsx3
                    import pyttsx3

                    engine = pyttsx3.init()
                    engine.save_to_file(text, audio_file)
                    engine.runAndWait()

                # Play audio
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                # Optional: Clean up audio file
                if not self.args.keep_audio:
                    try:
                        os.remove(audio_file)
                    except:
                        pass

            except queue.Empty:
                # No new descriptions
                pass
            except Exception as e:
                print(f"Error in speech synthesis: {e}")
                time.sleep(0.1)

        print("Speech synthesis stopped")

    def display_frames(self):
        """Optional: Display camera feed with current description"""
        print("Starting display...")

        while self.running:
            if not self.frame_queue.empty():
                try:
                    # Get a copy of the latest frame without removing it from the queue
                    frame = self.frame_queue.queue[0].copy()

                    # Add description text
                    if self.last_description:
                        # Wrap text to fit on screen
                        y_pos = 30
                        wrapped_text = self._wrap_text(
                            self.last_description, frame.shape[1] - 20, 30
                        )
                        for line in wrapped_text:
                            cv2.putText(
                                frame,
                                line,
                                (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )
                            y_pos += 30

                    # Display frame
                    cv2.imshow("Environment Description System", frame)

                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):  # ESC or 'q'
                        self.stop()

                except Exception as e:
                    print(f"Display error: {e}")

        cv2.destroyAllWindows()
        print("Display stopped")

    def _wrap_text(self, text, max_width, font_size):
        """Wrap text to fit on screen"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            # Rough estimate of text width
            text_size = len(test_line) * (font_size / 2)

            if text_size < max_width:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def start(self):
        """Start all processing threads"""
        self.running = True

        # Create and start threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.speech_thread = threading.Thread(target=self.speak_descriptions)

        self.capture_thread.start()
        self.process_thread.start()
        self.speech_thread.start()

        # Only run display in main thread if enabled
        if self.args.display:
            self.display_frames()
        else:
            # Wait for keyboard interrupt
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received")
                self.stop()

    def stop(self):
        """Stop all processing threads"""
        self.running = False

        # Wait for threads to finish
        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join()

        if hasattr(self, "process_thread") and self.process_thread.is_alive():
            self.process_thread.join()

        if hasattr(self, "speech_thread") and self.speech_thread.is_alive():
            self.speech_thread.join()

        # Release resources
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

        cv2.destroyAllWindows()
        print("System stopped")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Environment Description System for Jetson Nano"
    )

    # Camera settings
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")

    # Model settings
    parser.add_argument(
        "--vision_model",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="Hugging Face model for image captioning",
    )
    parser.add_argument(
        "--half_precision", action="store_true", help="Use FP16 for inference"
    )
    parser.add_argument(
        "--beam_size", type=int, default=3, help="Beam size for text generation"
    )

    # TTS settings
    parser.add_argument(
        "--tts_engine",
        type=str,
        default="coqui",
        choices=["coqui", "pyttsx3"],
        help="Text-to-speech engine to use",
    )
    parser.add_argument(
        "--keep_audio", action="store_true", help="Keep generated audio files"
    )

    # System settings
    parser.add_argument(
        "--cooldown",
        type=float,
        default=3.0,
        help="Minimum seconds between descriptions",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.3,
        help="Threshold for considering descriptions different (0-1)",
    )
    parser.add_argument(
        "--display", action="store_true", help="Display camera feed with descriptions"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    print("Initializing Environment Description System...")
    print(
        f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    # Initialize and start the system
    system = EnvironmentDescriptionSystem(args)

    try:
        system.start()
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    finally:
        system.stop()
        print("System shut down")


if __name__ == "__main__":
    main()
