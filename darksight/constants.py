"""Single source of truth for class ids, ExDark metadata codes and the COCO mapping."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# YOLO class ids 0..11 follow ExDark's official 1..12 order (imageclasslist.txt).
CLASS_NAMES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
NUM_CLASSES = len(CLASS_NAMES)

# Lower-cased aliases seen (or plausible) in ExDark annotation files -> class id.
CLASS_ALIASES = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}
CLASS_ALIASES.update({
    "bike": 0, "bicycles": 0,
    "boats": 1,
    "bottles": 2,
    "motorbikes": 9, "motorcycle": 9,
    "person": 10,
    "tables": 11, "diningtable": 11,
})

SPLITS = {1: "train", 2: "val", 3: "test"}

LIGHT_TYPES = {
    1: "Low", 2: "Ambient", 3: "Object", 4: "Single", 5: "Weak",
    6: "Strong", 7: "Screen", 8: "Window", 9: "Shadow", 10: "Twilight",
}
INDOOR_OUTDOOR = {1: "Indoor", 2: "Outdoor"}

# Ultralytics COCO-80 index -> ExDark class id. Names are asserted at runtime.
COCO_TO_EXDARK = {
    1: 0,    # bicycle
    8: 1,    # boat
    39: 2,   # bottle
    5: 3,    # bus
    2: 4,    # car
    15: 5,   # cat
    56: 6,   # chair
    41: 7,   # cup
    16: 8,   # dog
    3: 9,    # motorcycle
    0: 10,   # person
    60: 11,  # dining table
}
COCO_NAMES_EXPECTED = {
    1: "bicycle", 8: "boat", 39: "bottle", 5: "bus", 2: "car", 15: "cat",
    56: "chair", 41: "cup", 16: "dog", 3: "motorcycle", 0: "person", 60: "dining table",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_ZERODCE_WEIGHTS = PROJECT_ROOT / "weights" / "zerodce_epoch99.pth"
