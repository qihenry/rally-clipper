# Ping Pong Clipper (Modularized)

## Structure

- `main.py` - CLI entry point
- `pingpong_clipper/models.py` - dataclasses and config
- `pingpong_clipper/video_io.py` - ffmpeg checks and video frame loading
- `pingpong_clipper/roi.py` - editable ROI selector
- `pingpong_clipper/analysis.py` - mask building, motion scoring, smoothing, segment detection
- `pingpong_clipper/export.py` - clip export and debug file writing
- `pingpong_clipper/browser.py` - clip browser and playback
- `pingpong_clipper/pipeline.py` - orchestration

## Run

```bash
python main.py your_video.mp4
```


# Ping Pong Point Clipper

This is a practical MVP for clipping long ping pong recordings into **separate point videos**.

It does **not** try to detect the serve directly. Instead, it detects when the ball is likely already in play based on motion around the table, then shifts the clip earlier by default so the serve is still included.

## What it does

- takes one full ping pong video
- lets you manually mark the table area once
- detects sustained rally motion near the table
- adds **2 seconds before** the detected point and **2 seconds after** the end
- exports one MP4 file per point
- writes a CSV manifest with clip timestamps

## Why this version is built this way

This is the fastest realistic first build because of the following assumptions:

- Camera is usually stable on a tripod
- Audio is noisy in a club, so audio detection is weak
- Explicit serve detection is harder and less reliable than backtracking a little before “ball in play”. Camera angle will hide the ball sometimes
- manual table selection makes the detection much more stable and will remove interference from other players

## Requirements

- Python 3.10+
- `ffmpeg` installed and available on PATH
- Python packages:
  - `opencv-python`
  - `numpy`

Install packages:

```bash
pip install opencv-python numpy
```

Check ffmpeg:

```bash
ffmpeg -version
```
https://www.gyan.dev/ffmpeg/builds/#:~:text=ffmpeg%2Drelease%2Dessentials.zip,.ver%20.sha256

^ thats where I downloaded ffmpeg. then to set it to env path:
1. Press Windows key
2. Search: Environment Variables
3. Click Edit the system environment variables
4. Click Environment Variables
5. Under System variables → find Path
6. Click Edit
7. Click New
   Add: `C:\ffmpeg\bin`
8. Click OK on everything
## Run it

### Basic use

```bash
python pingpong_clipper.py /path/to/video.mp4
```

A frame will open. Draw a rectangle around the table and press **Enter** or **Space**.

The app will then create a folder named like:

```text
<video_name>_clips
```

Inside it, you will get files like:

```text
point_001.mp4
point_002.mp4
point_003.mp4
clips_manifest.csv
run_metadata.json
```

### Reuse a known table ROI (ignore for most part)

If you already know the rectangle, skip the selector:

```bash
python pingpong_clipper.py /path/to/video.mp4 --roi 320,210,980,360
```

That format is:

```text
x,y,w,h
```

## Useful tuning flags

### More context before each point

```bash
python pingpong_clipper.py input.mp4 --pre-roll 2.5
```

### Keep more after each point

```bash
python pingpong_clipper.py input.mp4 --post-roll 2.5
```

### Too many false clips?
Make detection stricter:

```bash
python pingpong_clipper.py input.mp4 --threshold-percentile 90 --threshold-floor 0.0045
```

### Missing real points?
Make detection looser:

```bash
python pingpong_clipper.py input.mp4 --threshold-percentile 80 --threshold-floor 0.0025
```

## Current detection logic

The detector uses frame-to-frame motion, focused around the table:

- table rectangle itself
- slightly expanded area around the table
- band above the table where the ball usually travels

It computes motion scores over time, smooths them, detects sustained “active” windows, merges nearby activity, and exports clips.


## Known limitations

This MVP can still make mistakes when:

- a player retrieves a ball near the table and creates motion that looks like play
- camera shakes
- the table region is selected too loosely or too tightly
- background movement is heavy
- the angle is extreme and the table is only partly visible

## Recommended next upgrades

Good next steps would be:

1. preview timeline before export
2. let the user merge/split clips manually
3. add a simple GUI instead of OpenCV ROI selection
4. add optional small-object tracking above the table
5. save a reusable profile per camera setup
6. add batch processing for many videos

## Output notes

The clips are re-encoded with H.264 + AAC for compatibility.

If you want faster exports later, the exporter can be changed to a stream-copy workflow, but this version prioritizes reliable cuts.
