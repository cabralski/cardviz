#    ______               ___    ___    
#   / ____/___ __________/ / |  / (_)___
#  / /   / __ `/ ___/ __  /| | / / /_  /
# / /___/ /_/ / /  / /_/ / | |/ / / / /_
# \____/\__,_/_/   \__,_/  |___/_/ /___/
									  
# Change this name to your `.pt` file.
MODEL_PATH = "YOLOv8 - PlayingCards.pt"

# IMPORTANT: Set this to your VideoCapture index!
# This must be an integer.
CAPTURE = 0

# Set this to your VideoCapture resolution.
CAPTURE_RESOLUTION = (1280 // 1.25, 960 // 1.25)

# Set this to your region of interest (rectangle).
# Use (0, -1) to get everything.
#           (Y₁, Y₂) (X₁, X₂)
CAMERA_ROI = (140, 500), (0, -1)

#    ______          __     __
#   / ____/___  ____/ /__  / /
#  / /   / __ \/ __  / _ \/ / 
# / /___/ /_/ / /_/ /  __/_/  
# \____/\____/\__,_/\___(_)   
							
# I love regular expressions.
import re

# Built-ins.
import os

# I love static typing.
from typing import Dict, Tuple

# Enumerable types.
from enum import Enum

class ANSIColor(Enum):
	"""An enumerable type of ANSI colors."""

	# The basic colors.
	RED = "\033[0;31m"
	GREEN = "\033[0;32m"
	BROWN = "\033[0;33m"
	BLUE = "\033[0;34m"
	PURPLE = "\033[0;35m"
	CYAN = "\033[0;36m"
	YELLOW = "\033[1;33m"
	
	# Some text styles.
	BOLD = "\033[1m"
	ITALIC = "\033[3m"
	CROSSED = "\033[9m"
	UNDERLINE = "\033[4m"
	
	# The "reset" color.
	END = "\033[0m"

def colored(text: str) -> str:
	"""
	Parse the text for ANSI color and style tags, and replace them
	with the corresponding ANSI sequences.
	"""
	
	# Regular expression to match <tag> and </tag>.
	tag_pattern = re.compile(r'<(/?)(\w+)>')
	
	def replace_tag(match: re.Match):

		# Extract tag type and name.
		tag_type, tag_name = match.groups()
		
		# Check if it's an open or close tag.
		if tag_type == "":
			return getattr(ANSIColor, tag_name.upper(), ANSIColor.END).value
		else: 
			return ANSIColor.END.value
	
	# Substitute the tags with ANSI codes.
	result = tag_pattern.sub(replace_tag, text)
	return result

def card_to_value(card: str) -> int:
	
	# Remove the suit symbol, if it has.
	card = card.replace("♣", "").replace("♦", "").replace("♥", "").replace("♠", "")

	# J, K, and Q corresponds to 10.
	if card in ["J", "Q", "K"]:
		return 10

	# "A" corresponds to 1.
	if card == 'A':
		return 1
	
	# Otherwise, just return the rank.
	return int(card)

# This is a list of packages that need to be installed.
MISSING_PACKAGES = []

try:
	
	import numpy as np

except:
	
	print(colored("[<red>ERROR</red>] NumPy is <bold>not</bold> installed."))
	MISSING_PACKAGES += ["numpy"]

try:
	
	import cv2 as ocv

except:
	
	print(colored("[<red>ERROR</red>] OpenCV is <bold>not</bold> installed."))
	MISSING_PACKAGES += ["opencv-python"]

try:
	
	from sklearn.cluster import DBSCAN

except:

	print(colored("[<red>ERROR</red>] Scikit-Learn is <bold>not</bold> installed."))
	MISSING_PACKAGES += ["scikit-learn"]

try:
	
	from ultralytics import YOLO

except:
	
	print(colored("[<red>ERROR</red>] Ultralytics is <bold>not</bold> installed."))
	MISSING_PACKAGES += ["ultralytics"]

# If there are one or more packages pending installation, print them.
if len(MISSING_PACKAGES) >= 1:
	
	print("Make sure to install the required packages by running the following command:")
	print(colored(f"<green>>>></green> pip install {' '.join(MISSING_PACKAGES)}"))
	exit(1)

try:
	
	print(colored(f"[<blue>INFO</blue>] Loading model \"{os.path.basename(MODEL_PATH)}\"."))
	model = YOLO(MODEL_PATH)

except:
	
	print(colored("[<red>ERROR</red>] Couldn't find the model in the given path."))
	print(colored(f"Make sure to include the <green>{os.path.basename(MODEL_PATH)}</green> model in the current directory."))
	exit(1)

try:

	print(colored(f"[<blue>INFO</blue>] Opening VideoCapture({CAPTURE})..."))
	capture = ocv.VideoCapture(CAPTURE)

	# If the VideoCapture couldn't be opened, raise an exception.
	if not capture.isOpened():
		raise

except:

	print(colored(f"[<red>ERROR</red>] Couldn't open the VideoCapture({CAPTURE}), try changing the index to {CAPTURE + 1}."))
	exit(1)

# Set VideoCapture resolution.
print(colored(f"[<blue>INFO</blue>] Setting VideoCapture resolution.."))
capture.set(ocv.CAP_PROP_FRAME_WIDTH, CAPTURE_RESOLUTION[0])
capture.set(ocv.CAP_PROP_FRAME_HEIGHT, CAPTURE_RESOLUTION[1])

while True:
	
	# Capture the video frame.
	rectangle, frame = capture.read() 
  
	# If there isn't a rectangle (or a frame), skip.
	if not rectangle:
		continue

	# Extract the region of interest.
	frame = frame[CAMERA_ROI[0][0] : CAMERA_ROI[0][1], CAMERA_ROI[1][0] : CAMERA_ROI[1][1]]

	# Normalize the frame.
	# frame = ocv.normalize(frame, frame, 0, 255, ocv.NORM_MINMAX)

	# Infer using the YOLOv8 model.
	clusters = model(frame, verbose = False, imgsz = 1024, iou = 0.0, agnostic_nms = True)[0]

	# This is a list of every card centroid.
	card_centroid: Dict[str, Tuple[int, int]] = {}

	for annotation in clusters:

		# This is the predicted class as an integer from 0 to 51 (52 cards total).
		predicted = int(annotation.boxes.cls[0])
		
		# This is a full label, such as "A♣".
		label = annotation.names[predicted]

		# This is just the rank of the card (2, 3, ..., K, A).
		rank = annotation.names[predicted].replace("♣", "").replace("♦", "").replace("♥", "").replace("♠", "")

		# This is just the suit of the card (♣, ♦, ♥, ♠).
		suit = annotation.names[predicted][-2 : -1]

		# This is the bounding box (make sure to convert to integer).
		x1, y1, x2, y2 = [int(tensor) for tensor in annotation.boxes.xyxy[0]]

		# This is the centroid of the bounding box.
		cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

		# The suits "♣" and "♠" are black, "♦" and "♥" are red.
		suit_color = (0, 0, 0)
		if ("♦" in label) or ("♥" in label):
			suit_color = (0, 0, 255)

		# The suits "♣" and "♠" have white text, "♦" and "♥" have red text.
		rank_color = (255, 255, 255)
		if ("♦" in label) or ("♥" in label):
			rank_color = (0, 0, 255)

		# Set the card position.
		card_centroid[label] = (cx, cy)

		# Draw the rectangle.
		ocv.rectangle(frame, (x1, y1), (x2, y2), suit_color, 1)

		# Draw a line to (0, 0).
		# ocv.line(frame, (cx, cy), (0, 0), (255, 255, 255), 1)

		# Calculate the position for the text.
		ocv.putText(frame, rank, (x1, y1 - 10), ocv.FONT_HERSHEY_COMPLEX, 0.8, rank_color, 1)

	# If there are more than 2 cards..
	if len(card_centroid) >= 2:
		
		# These are the card values.
		card_values = {label: card_to_value(label) for label in list(card_centroid.keys())}

		# Do the clustering using DBSCAN.
		clustering = DBSCAN(eps = 128, min_samples = 1).fit(np.array(list(card_centroid.values())))

		# Extract the indices.
		indices = clustering.labels_

		# Segment the clusters based on the "indices".
		clusters = {}
		for element, mask in zip(list(card_centroid.keys()), indices):
			if mask not in clusters:
				clusters[mask] = []
			clusters[mask].append(element)

		# Draw the sums above the clusters.
		for cluster in clusters:

			# Here are the cards of this cluster.
			cluster_cards = clusters[cluster]

			# If the cluster only has one card, skip.
			if len(cluster_cards) <= 1:
				continue

			# This is the sum of the cluster.
			cluster_sum = sum([card_to_value(card) for card in cluster_cards])

			# This is the cluster centroid.
			kx, ky = np.mean([card_centroid[card] for card in cluster_cards], axis = 0).astype(int).tolist()

			# Draw a line to the centroid.
			for card in cluster_cards:
				ocv.line(frame, card_centroid[card], (kx, ky - (80)), (255, 255, 255), 1)

			# Display the sum.
			ocv.putText(frame, f"{cluster_sum}", (kx, ky - (80)), ocv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)

	# Display the resulting frame.
	ocv.imshow('CardViz', frame)

	# If the user presses "q", exit the inference loop.
	if (ocv.waitKey(1) & 0xFF) == ord('q'):
		print(colored(f"[<blue>INFO</blue>] Closing VideoCapture and stopping inference."))
		break

# Release the VideoCapture object!
capture.release()

# Also, destroy all windows.
ocv.destroyAllWindows()