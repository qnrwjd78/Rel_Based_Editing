#!/usr/bin/env python3
import json


def scale_bbox(b, scale):
    x1, y1, x2, y2 = b
    return [
        round(x1 * scale),
        round(y1 * scale),
        round(x2 * scale),
        round(y2 * scale),
    ]


def main():
    scale = 512 / 776
    with open("condition/test_condition1.json", "r", encoding="utf-8") as f:
        cond = json.load(f)

    for k, v in list(cond.items()):
        if isinstance(v, list) and len(v) == 4:
            cond[k] = scale_bbox(v, scale)

    with open("condition/test_condition1_512.json", "w", encoding="utf-8") as f:
        json.dump(cond, f, indent=4)
    print("saved condition/test_condition1_512.json")


if __name__ == "__main__":
    main()
