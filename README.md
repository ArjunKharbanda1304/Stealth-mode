# Stealth-mode


# Player Re-Identification in Sports Footage

## Overview
This project implements player re-identification in a single camera feed using YOLOv11 and DeepSORT.

## How to Run
1. Install dependencies:

2. Place the provided model weights and video in the project folder.

3. Run the code:

4. The output video will be saved as `output_with_tracking.mp4`.

## Approach
- YOLOv11 detects players frame by frame.
- DeepSORT assigns consistent IDs to players and handles re-identification.

## Challenges
- Handling occlusion and players overlapping.
- Ensuring IDs remain consistent when players leave and re-enter.

## Future Improvements
- Use a stronger appearance embedding network for even better re-identification.
