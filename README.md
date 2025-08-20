Fruit Ninja Python Game

This is a Python-based version of the popular Fruit Ninja game, created using OpenCV for webcam access, MediaPipe for hand tracking, and Pygame for rendering the game. The game uses your webcam feed as the background, and you slice fruits and avoid bombs with your finger movements.

Features

Webcam Integration: Uses MediaPipe’s Hand Tracking to detect fingertip movements and trigger slices.

Full-Screen Webcam Feed: The game runs in full-screen mode with the webcam as the background.

Fruit Spawning: Fruits spawn randomly and fall from the top of the screen.

Bomb Interaction: Slicing a bomb triggers a flashbang and screen shake effect.

Slicing Visuals: Bright red slash lines with a glowing effect when slicing fruits.

Interactive: Use your index finger to slice fruits by swiping across the screen.

Requirements

Make sure you have Python 3.10+ installed, then set up the environment with the following dependencies:

1. Clone the repository
git clone https://github.com/yourusername/fruit-ninja-python.git
cd fruit-ninja-python

2. Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt


Where requirements.txt contains:

opencv-python
mediapipe
pygame
numpy

4. Run the game

After setting up the environment, run the following command to start the game:

python fruitninja_game.py


The game will launch with your webcam feed as the background, and you can start playing by swiping your index finger to slice fruits!

How to Play

Start the game: The game will automatically start after loading the webcam.

Slice fruits: Use your index finger to swipe and slice fruits.

Avoid bombs: If you slice a bomb, the screen will shake, and a flashbang effect will appear.

Goal: Slice as many fruits as possible before the time runs out (1 minute).

Restart: Press R to restart the game at any time.

Quit: Press Q or ESC to quit the game.

Game Controls

Left-click or touch to swipe and slice fruits.

R to restart the game.

Q or ESC to quit the game.

F to toggle fullscreen mode.

Technology Stack

Python 3.x

OpenCV for webcam access and drawing.

MediaPipe for hand tracking and finger detection.

Pygame for game rendering and handling the game loop.

NumPy for numerical operations.

License

This project is licensed under the MIT License - see the LICENSE
 file for details.

Known Issues

Some devices may have lag or frame drops depending on the camera quality and system resources.

Low FPS might occur on low-end devices due to hand tracking and real-time video rendering.

Future Enhancements

Web Version: A browser-based version using HTML5, JavaScript, and MediaPipe for fingertip tracking.

Leaderboard: Track high scores and best times for players.

This version is for local play with Python—you will need to install Python and set up the environment to run the game.
