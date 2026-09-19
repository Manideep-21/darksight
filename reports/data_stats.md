# ExDark → YOLO conversion report

Settings: max_side=1280, min_box=2.0px, exif=auto, limit=none

## Images and boxes per split

| split | images | boxes |
|---|---|---|
| train | 3000 | 10381 |
| val | 1800 | 6212 |
| test | 2563 | 7117 |
| **total** | 7363 | 23710 |

## Boxes per class

| id | class | train | val | test | total |
|---|---|---|---|---|---|
| 0 | Bicycle | 463 | 239 | 418 | 1120 |
| 1 | Boat | 530 | 344 | 515 | 1389 |
| 2 | Bottle | 654 | 506 | 433 | 1593 |
| 3 | Bus | 333 | 209 | 164 | 706 |
| 4 | Car | 1238 | 770 | 919 | 2927 |
| 5 | Cat | 311 | 173 | 425 | 909 |
| 6 | Chair | 1138 | 630 | 609 | 2377 |
| 7 | Cup | 864 | 436 | 356 | 1656 |
| 8 | Dog | 335 | 193 | 490 | 1018 |
| 9 | Motorbike | 490 | 340 | 242 | 1072 |
| 10 | People | 3271 | 1954 | 2235 | 7460 |
| 11 | Table | 754 | 418 | 311 | 1483 |

## Images per lighting condition

| light | train | val | test |
|---|---|---|---|
| Low | 51 | 56 | 134 |
| Ambient | 716 | 457 | 925 |
| Object | 235 | 119 | 144 |
| Single | 316 | 191 | 255 |
| Weak | 298 | 175 | 232 |
| Strong | 999 | 522 | 538 |
| Screen | 69 | 68 | 28 |
| Window | 141 | 103 | 119 |
| Shadow | 28 | 14 | 14 |
| Twilight | 147 | 95 | 174 |

## Data-quality log

- listed in imageclasslist but image file missing: **0** []
- image files not in imageclasslist (ignored): **0** []
- unreadable images: **0** []
- images without annotation file: **0** []
- images with zero boxes after conversion: **1** ['2015_05894']
- boxes read: 23710, kept: 23710, clipped to image: 50, dropped (<2.0px): 0, unknown class: {}
- source colour modes: {'RGB': 7263, 'P': 1, 'L': 35, 'RGBA': 64}
- images downscaled to max side 1280: 526
- images with EXIF rotation tag: **1** ['2015_05337']; kept un-rotated because boxes match raw pixels: ['2015_05337']
- images whose boxes only fit the *un-rotated* frame (check these!): **0** []
