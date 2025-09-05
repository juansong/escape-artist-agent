#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import imageio.v2 as imageio

def main():
    ap = argparse.ArgumentParser(description="Make a hero GIF from rollout PNGs")
    ap.add_argument("--frames", nargs="+", required=True,
                    help="Ordered list of PNG files (e.g., easy/medium/hard rollouts).")
    ap.add_argument("--out", type=str, default="assets/escape-artist-hero.gif",
                    help="Output GIF path.")
    ap.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2).")
    args = ap.parse_args()

    images = []
    for f in args.frames:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Missing frame: {p}")
        images.append(imageio.imread(p))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, images, fps=args.fps, loop=0)
    print(f"Saved GIF: {out_path}")

if __name__ == "__main__":
    main()
