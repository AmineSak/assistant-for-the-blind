"""
Simplified real-time environment description system using BLIP vision model
"""

import platform
import queue
import threading
import time

import cv2
import pyttsx3
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class RealTimeDescriber:
    def __init__(self):
        # Initialize state variables first
        self.running = False
        self.current_frame = None
        self.last_description = ""
        self.last_description_time = 0
        self.description_cooldown = 3.0  # seconds between descriptions
        self.tts_queue = queue.Queue()
        self.tts_thread = None

        # Now setup components
        self.setup_model()
        self.setup_camera()
        self.setup_tts()

    def setup_model(self):
        """Initialize BLIP model"""
        print("Loading BLIP model...")
        model_name = "Salesforce/blip-image-captioning-base"

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

        # Use GPU if available
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def setup_camera(self):
        """Initialize camera"""
        print("Setting up camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        print("Camera ready")

    def setup_tts(self):
        """Initialize text-to-speech"""
        print("Setting up text-to-speech...")
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", 150)
            self.tts_available = True
            print("TTS ready")
        except Exception as e:
            print(f"TTS setup failed: {e}")
            self.tts_available = False

    def _process_tts_queue(self):
        """Process TTS queue in a separate thread"""
        while self.running:
            try:
                # Get text from queue with timeout to allow checking running status
                text = self.tts_queue.get(timeout=1.0)
                if text is not None:
                    print(f"Speaking: {text}")
                    try:
                        # Use a separate TTS engine instance for thread safety
                        if platform.system().lower() == "darwin":
                            engine = pyttsx3.init("nsss")
                        else:
                            engine = pyttsx3.init()

                        engine.setProperty("rate", 150)
                        engine.say(text)
                        engine.runAndWait()

                        # No need to call engine.stop() after runAndWait()
                        # runAndWait() already handles the engine lifecycle

                    except Exception as e:
                        print(f"TTS speech error: {e}")
                        # Try with the main engine as fallback
                        try:
                            if self.tts_available and hasattr(self, "tts_engine"):
                                self.tts_engine.say(text)
                                self.tts_engine.runAndWait()
                        except Exception as fallback_error:
                            print(f"TTS fallback also failed: {fallback_error}")

                    self.tts_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS queue processing error: {e}")
                continue

    def capture_frame(self):
        """Capture a single frame from camera"""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
        return ret

    def generate_description(self, frame):
        """Generate description for the current frame"""
        try:
            # Convert frame to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Process with BLIP using inference mode
            with torch.inference_mode():
                # Process image - let the processor handle tensor shapes properly
                inputs = self.processor(images=pil_image, return_tensors="pt").to(
                    self.device
                )
                print(f"Inputs: {len(inputs)}")

                # Generate description
                outputs = self.model.generate(
                    **inputs,
                    # max_length=50,
                    num_beams=5,
                    do_sample=False,
                    early_stopping=True,
                )

                description = self.processor.decode(
                    outputs[0], skip_special_tokens=True
                )

            return description.strip()

        except Exception as e:
            print(f"Error generating description: {e}")
            return None

    def is_description_different(self, new_desc):
        """Check if new description is significantly different from the last one"""
        if not self.last_description:
            return True

        # Simple word-based difference check
        old_words = set(self.last_description.lower().split())
        new_words = set(new_desc.lower().split())

        # Calculate overlap ratio
        overlap = len(old_words.intersection(new_words))
        total_unique = len(old_words.union(new_words))

        if total_unique == 0:
            return False

        similarity = overlap / total_unique
        return similarity < 0.7  # Consider different if less than 70% similar

    def speak_text(self, text):
        """Add text to TTS queue"""
        if not self.tts_available:
            print("TTS not available")
            return

        try:
            print(f"Queueing text for TTS: {text}")
            self.tts_queue.put(text, timeout=1.0)
        except queue.Full:
            print("TTS queue is full, skipping this description")
        except Exception as e:
            print(f"Error queueing TTS: {e}")

    def process_and_describe(self):
        """Main processing loop for generating descriptions"""
        while self.running:
            current_time = time.time()

            # Check cooldown period
            if current_time - self.last_description_time < self.description_cooldown:
                time.sleep(0.1)
                continue

            if self.current_frame is not None:
                # Generate description
                description = self.generate_description(self.current_frame)

                if description and self.is_description_different(description):
                    print(f"Description: {description}")

                    # Update tracking variables
                    self.last_description = description
                    self.last_description_time = current_time

                    # Speak the description
                    self.speak_text(description)

            time.sleep(0.1)

    def display_video(self):
        """Display video feed with overlay text"""
        while self.running:
            if not self.capture_frame():
                print("Failed to capture frame")
                time.sleep(0.1)
                continue

            # Create display frame
            display_frame = self.current_frame.copy()

            # Add description text overlay
            if self.last_description:
                # Wrap text for display
                words = self.last_description.split()
                lines = []
                current_line = []
                max_chars_per_line = 50

                for word in words:
                    test_line = " ".join(current_line + [word])
                    if len(test_line) <= max_chars_per_line:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]

                if current_line:
                    lines.append(" ".join(current_line))

                # Draw text lines
                y_offset = 30
                for line in lines:
                    cv2.putText(
                        display_frame,
                        line,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    y_offset += 30

            # Show frame
            cv2.imshow("Real-time Environment Description", display_frame)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                self.stop()
                break

    def start(self):
        """Start the real-time description system"""
        print("Starting real-time environment describer...")
        print("Press 'q' or ESC to quit")

        self.running = True

        # Start TTS processing thread
        if self.tts_available:
            self.tts_thread = threading.Thread(
                target=self._process_tts_queue, daemon=True
            )
            self.tts_thread.start()
            print("TTS thread started")

        # Start description processing in background thread
        self.process_thread = threading.Thread(
            target=self.process_and_describe, daemon=True
        )
        self.process_thread.start()

        # Run video display in main thread
        self.display_video()

    def stop(self):
        """Stop the system"""
        print("Stopping...")
        self.running = False

        # Wait for TTS queue to empty with timeout
        if self.tts_available and hasattr(self, "tts_queue"):
            try:
                # Add a None sentinel to signal the TTS thread to stop
                self.tts_queue.put(None, timeout=1.0)
                # Wait a bit for the queue to process
                time.sleep(1.0)
            except Exception as e:
                print(f"Error while stopping TTS: {e}")

        if hasattr(self, "cap"):
            self.cap.release()

        cv2.destroyAllWindows()
        print("Stopped")


def main():
    """Main function"""
    print("Real-time Environment Description System")
    print("=" * 40)

    try:
        describer = RealTimeDescriber()
        describer.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Shutting down...")


if __name__ == "__main__":
    main()
