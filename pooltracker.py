import cv2 as cv
import numpy as np
import scipy as sc
import os
import matplotlib.pyplot as plt

# Change path of the execution ot the path of file's directory 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system('cls')

# === CONSTANTS ===
NB_EMPTY = 50
NB_FULL = 900
MAX_FULL_WHITE = 560
MIN_FULL_BLACK = 45

DRAW_WHITE_BALL   = (0, 255, 255)
DRAW_BLACK_BALL   = (0, 0, 0)
DRAW_FULL_BALL    = (255, 0, 0)
DRAW_STRIPED_BALL = (0, 255, 0)

BOARD_TOP_LEFT = (321, 458)
BOARD_BOTTOM_RIGHT = (2037, 1354)


def rescaleFrame(frame, scale=0.75):
    # For images, videos and live videos
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load the image
image = cv.imread('test_board.png')

# Mask
blank = np.zeros(image.shape[:2], dtype='uint8') #important : mask needs to be the same size of the image
mask = cv.rectangle(blank, BOARD_TOP_LEFT, BOARD_BOTTOM_RIGHT, 255, thickness=cv.FILLED)
image = cv.bitwise_and(image, image, mask=mask)

def get_pixels_in_circle(image, center, radius):
    """
    Extracts all pixels inside a circle from the image.

    Parameters:
    image (ndarray): The input image.
    center (tuple): The (x, y) center of the circle.
    radius (int): The radius of the circle.

    Returns:
    ndarray: A 1D array of pixel values inside the circle.
    """
    # Create a mask with the same size as the image
    mask = np.zeros(image.shape[:2], dtype='uint8')

    # Draw a filled circle on the mask
    cv.circle(mask, center, radius, 255, -1)

    # Apply the mask to the image
    masked_image = cv.bitwise_and(image, image, mask=mask)

    # Extract the pixels inside the circle
    pixels_in_circle = masked_image[mask == 255]

    return pixels_in_circle

#==============================================================================================================
# BALL CLASS

class Ball:
    def __init__(self, id, x, y, radius):
        self.id = id
        self.x = x  # x-coordinate of the ball's center
        self.y = y  # y-coordinate of the ball's center
        self.radius = radius  # Radius of the ball
        self.label = 1  # Classification label: 0 (full), 1 (striped), 2 (white ball), 3 (black ball)
        self.nb_white = 0.0  # Percentage of white pixels in the circle
        self.nb_black = 0.0  # Percentage of black pixels in the circle

    def analyze(self, image):
        """Analyze the ball region to compute teh number of black and white pixels."""
        # Extract the square region of interest
        # Create a mask with the same size as the image
        mask = np.zeros(image.shape[:2], dtype='uint8')

        # Draw a filled circle on the mask
        cv.circle(mask, (self.x,self.y), self.radius, 255, -1)

        # Apply the mask to the image
        masked_image = cv.bitwise_and(image, image, mask=mask)

        # Extract the pixels inside the circle
        cirlce_roi = masked_image[mask == 255]
        
        # Count black and white
        self.nb_black = np.sum(np.all(cirlce_roi < [50, 50, 50], axis=-1))
        self.nb_white = np.sum(np.all(cirlce_roi > [200, 200, 200], axis=-1))

    def classify(self):
        """Classify the ball based on its attributes."""
        if self.nb_white > NB_FULL:  # White ball
            self.label = 2
        elif self.nb_black > NB_FULL:  # Black ball
            self.label = 3
        elif self.nb_white < NB_EMPTY and self.nb_black < NB_EMPTY:  # Full ball
            self.label = 0
        elif self.nb_white < MAX_FULL_WHITE and self.nb_black > MIN_FULL_BLACK:  # Full ball
            self.label = 0
        else:  # Striped ball
            self.label = 1

    def __str__(self):
        return (f"Ball {self.id}: black {self.nb_black} white {self.nb_white}")


#==============================================================================================================

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv.GaussianBlur(gray, (9, 9), 2)

# Use Hough Circle Transform to detect circles
circles = cv.HoughCircles(
    blurred, 
    cv.HOUGH_GRADIENT, 
    dp=1.2, 
    minDist=30, 
    param1=50, 
    param2=30, 
    minRadius=23, 
    maxRadius=23
)

# Initialize the result array
result_array = []

# Process each detected circle
if circles is not None:
    circles = np.uint16(np.around(circles))
    balls = []
    for i in range(len(circles[0, :])):
        circle = circles[0,i]
        x, y, radius = circle

        # Create a Ball instance
        ball = Ball(i, x, y, radius)

        # Analyze the ball
        ball.analyze(image)

        # Classify the ball
        ball.classify()

        # Draw the ball with a specific color based on its label
        if ball.label == 0:  # Full ball
            cv.circle(image, (ball.x, ball.y), ball.radius, DRAW_FULL_BALL, 2)
        elif ball.label == 1:  # Striped ball
            cv.circle(image, (ball.x, ball.y), ball.radius, DRAW_STRIPED_BALL, 2)
        elif ball.label == 2:  # White ball
            cv.circle(image, (ball.x, ball.y), ball.radius, DRAW_WHITE_BALL, 2)
        elif ball.label == 3:  # Black ball
            cv.circle(image, (ball.x, ball.y), ball.radius, DRAW_BLACK_BALL, 2)

        # Write the ball ID on the image
        text_position = (ball.x + ball.radius + 5, ball.y + ball.radius + 5)
        cv.putText(image, str(ball.id), text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Append to the list of balls
        balls.append(ball)

    # Print the details of all balls
    for ball in balls:
        print(ball)

#==============================================================================================================

# Resize the image
image = rescaleFrame(image, 0.645)

# Show result
cv.imshow('Board', image)

cv.waitKey(0)
cv.destroyAllWindows()