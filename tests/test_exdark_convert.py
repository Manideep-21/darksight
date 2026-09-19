import numpy as np
import pytest
from PIL import Image

from darksight.constants import CLASS_NAMES, PROJECT_ROOT
from darksight.data.exdark import (ConversionStats, RawBox, box_to_yolo, index_files, normalize_class,
                                   parse_bbgt, read_imageclasslist, yolo_to_xyxy)
from darksight.io_utils import load_rgb_with_info

SAMPLE = "% bbGt version=3\nBicycle 204 28 271 193 0 0 0 0 0 0 0\npeople 10 40 50 120 0 0 0 0 0 0 0\n\n"


def test_parse_bbgt_skips_header_and_blank_lines():
    boxes = parse_bbgt(SAMPLE)
    assert [b.class_name for b in boxes] == ["Bicycle", "people"]
    assert (boxes[0].left, boxes[0].top, boxes[0].width, boxes[0].height) == (204, 28, 271, 193)


def test_class_ids_follow_official_order():
    assert CLASS_NAMES[0] == "Bicycle" and CLASS_NAMES[10] == "People" and CLASS_NAMES[11] == "Table"
    assert normalize_class(" PEOPLE ") == 10
    assert normalize_class("Motorbike") == 9
    assert normalize_class("unicorn") is None


def test_yolo_round_trip():
    w, h = 640, 480
    y = box_to_yolo(RawBox("Car", 100, 50, 200, 100), w, h)
    cls, xc, yc, bw, bh = y
    assert cls == 4
    assert (xc, yc, bw, bh) == pytest.approx((200 / 640, 100 / 480, 200 / 640, 100 / 480))
    assert yolo_to_xyxy(xc, yc, bw, bh, w, h) == pytest.approx((100, 50, 300, 150))


def test_clipping_and_small_box_drop():
    stats = ConversionStats()
    y = box_to_yolo(RawBox("Dog", -10, 400, 100, 200), 640, 480, stats=stats)  # sticks out left and bottom
    assert y is not None
    x1, y1, x2, y2 = yolo_to_xyxy(*y[1:], 640, 480)
    assert (x1, y1, x2, y2) == pytest.approx((0, 400, 90, 480))
    assert box_to_yolo(RawBox("Dog", 700, 10, 50, 50), 640, 480, stats=stats) is None  # fully outside
    assert box_to_yolo(RawBox("Cup", 10, 10, 1, 30), 640, 480, stats=stats) is None    # too thin
    assert stats.clipped == 2 and stats.dropped_small == 2 and stats.boxes_out == 1


def test_unknown_class_counted():
    stats = ConversionStats()
    assert box_to_yolo(RawBox("Horse", 1, 1, 10, 10), 100, 100, stats=stats) is None
    assert stats.unknown_classes == {"Horse": 1}


def test_official_split_counts():
    recs = read_imageclasslist(PROJECT_ROOT / "configs" / "exdark_imageclasslist.txt")
    assert len(recs) == 7363
    counts = {s: sum(r.split == s for r in recs) for s in ("train", "val", "test")}
    assert counts == {"train": 3000, "val": 1800, "test": 2563}
    assert {r.class_id for r in recs} == set(range(12))
    assert len({r.stem.lower() for r in recs}) == 7363


def test_index_files_case_insensitive_and_duplicate_detection(tmp_path):
    (tmp_path / "Car").mkdir()
    (tmp_path / "Car" / "2015_00001.JPG").write_bytes(b"x")
    (tmp_path / "Car" / "2015_00001.jpg.txt").write_text(SAMPLE)
    imgs = index_files(tmp_path, lambda p: p.suffix.lower() == ".jpg")
    anns = index_files(tmp_path, lambda p: p.suffix == ".txt")
    assert set(imgs) == set(anns) == {"2015_00001"}
    (tmp_path / "Dog").mkdir()
    (tmp_path / "Dog" / "2015_00001.png").write_bytes(b"y")
    with pytest.raises(ValueError):
        index_files(tmp_path, lambda p: p.suffix.lower() in {".jpg", ".png"})


def test_exif_rotation_and_mode_conversion(tmp_path):
    arr = np.zeros((20, 40), dtype=np.uint8)  # grayscale, 40 wide x 20 high
    im = Image.fromarray(arr, mode="L")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    p = tmp_path / "rot.jpg"
    im.save(p, exif=exif)
    rotated, info = load_rgb_with_info(p, apply_exif=True)
    raw, _ = load_rgb_with_info(p, apply_exif=False)
    assert info["orientation"] == 6 and info["raw_size"] == (40, 20)
    assert rotated.shape == (40, 20, 3) and raw.shape == (20, 40, 3)
