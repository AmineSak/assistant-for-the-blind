#!/usr/bin/env python3
"""
Alternative implementation using a specific BLIP model with correct import structure
to address the feature extractor issue
"""

import argparse
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import pygame
import torch
from PIL import Image


class SimpleBLIPDescriber:
    def __init__(self):
        self.setup_model()
        self.setup_camera()
        self.setup_tts()
        self.frame_queue = queue.Queue(maxsize=2)
        self.text_queue = queue.Queue(maxsize=5)
        self.running = False
        self.last_description = ""
        self.last_description_time = 0
        pygame.mixer.init()

    def setup_model(self):
        print("Loading BLIP model...")

        # Import specific BLIP model classes
        from transformers import BlipForConditionalGeneration, BlipProcessor

        # Load model with specific class (not using AutoModel)
        model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set to evaluation modes
        self.model.eval()
        print(f"BLIP model loaded on {self.device}")

    def setup_camera(self):
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)  # Use camera 0
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        print("Camera initialized")

    def setup_tts(self):
        print("Setting up TTS (pyttsx3)...")
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)  # Speed
            print("TTS initialized")
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            self.engine = None

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
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass

    def process_frames(self):
        """Process frames with vision model to generate descriptions"""
        print("Starting frame processing...")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)

                # Check if cooldown period has passed
                current_time = time.time()
                if current_time - self.last_description_time < 3.0:  # 3 second cooldown
                    continue

                # Convert to PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Generate caption with BLIP
                with torch.no_grad():
                    # Process image
                    inputs = self.processor(pil_image, return_tensors="pt").to(
                        self.device
                    )

                    # Generate caption
                    output = self.model.generate(**inputs, max_length=50)
                    caption = self.processor.decode(output[0], skip_special_tokens=True)

                # Check if caption is different enough from last one
                if self._is_different_enough(caption):
                    print(f"New description: {caption}")
                    self.last_description = caption
                    self.last_description_time = current_time

                    # Add to speech queue
                    if not self.text_queue.full():
                        self.text_queue.put(caption)

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)

    def _is_different_enough(self, new_desc):
        """Simple check if new description is different from previous"""
        if not self.last_description:
            return True

        common_words = set(new_desc.lower().split()) & set(
            self.last_description.lower().split()
        )
        total_words = set(new_desc.lower().split()) | set(
            self.last_description.lower().split()
        )

        if not total_words:
            return False

        difference_ratio = 1 - (len(common_words) / len(total_words))
        return difference_ratio > 0.3  # Consider different if 30% words changed

    def speak_descriptions(self):
        """Speak the generated descriptions"""
        print("Starting speech synthesis...")
        output_dir = "generated_audio"
        os.makedirs(output_dir, exist_ok=True)

        while self.running:
            try:
                text = self.text_queue.get(timeout=1.0)

                if not text.strip():
                    continue

                # if self.engine:
                #     # Speak directly with pyttsx3
                #     self.engine.say(text)
                self.engine.runAndWait()
                if True:
                    # Fallback method - save to file and play with pygame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_file = os.path.join(output_dir, f"desc_{timestamp}.wav")

                    # Try importing pyttsx3 again
                    try:
                        import pyttsx3

                        temp_engine = pyttsx3.init()
                        temp_engine.save_to_file(text, audio_file)
                        temp_engine.runAndWait()

                        # Play with pygame
                        pygame.mixer.music.load(audio_file)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)

                        # Clean up
                        os.remove(audio_file)
                    except Exception as e:
                        print(f"Speech synthesis failed: {e}")

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in speech synthesis: {e}")
                time.sleep(0.1)

    def display_frames(self):
        """Show camera feed with descriptions"""
        print("Starting display...")

        while self.running:
            if not self.frame_queue.empty():
                try:
                    # Get a copy of the latest frame
                    frame = self.frame_queue.queue[0].copy()

                    # Add description text
                    if self.last_description:
                        # Split into lines
                        text = self.last_description
                        max_width = frame.shape[1] - 20
                        y_pos = 30

                        # Simple text wrapping
                        words = text.split()
                        lines = []
                        current_line = []

                        for word in words:
                            test_line = " ".join(current_line + [word])
                            if len(test_line) * 10 < max_width:  # Rough estimation
                                current_line.append(word)
                            else:
                                lines.append(" ".join(current_line))
                                current_line = [word]

                        if current_line:
                            lines.append(" ".join(current_line))

                        # Draw text lines
                        for line in lines:
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

                    # Display
                    cv2.imshow("Environment Description", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):  # ESC or 'q'
                        self.stop()

                except Exception as e:
                    print(f"Display error: {e}")

        cv2.destroyAllWindows()

    def start(self):
        """Start all threads"""
        self.running = True

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.speech_thread = threading.Thread(target=self.speak_descriptions)

        self.capture_thread.start()
        self.process_thread.start()
        self.speech_thread.start()

        # Run display in main thread
        self.display_frames()

    def stop(self):
        """Stop all threads"""
        self.running = False

        if hasattr(self, "capture_thread"):
            self.capture_thread.join()
        if hasattr(self, "process_thread"):
            self.process_thread.join()
        if hasattr(self, "speech_thread"):
            self.speech_thread.join()

        if hasattr(self, "cap"):
            self.cap.release()

        cv2.destroyAllWindows()
        print("System stopped")


def main():
    print("Starting simplified Environment Description System")
    print(
        f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    try:
        system = SimpleBLIPDescriber()
        system.start()
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if "system" in locals():
            system.stop()
        print("System shut down")


if __name__ == "__main__":
    main()
