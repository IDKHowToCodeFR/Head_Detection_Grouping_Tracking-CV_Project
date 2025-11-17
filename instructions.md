# Step-by-Step Instructions for Pipeline Execution

## Section 1: Create the Project Folders

Create a main directory (example: `Head_Detection_Grouping_Tracking`).
Inside it, create the following subfolders:

```
0_input_videos
models
```

The structure:

```
Head_Detection_Grouping_Tracking/
│
├── 0_input_videos/
├── models/
```

## Section 2: Create the Code Files

Inside the main folder, create four empty files:

```
config.py
utils.py
main.py
requirements.txt
```

Updated project layout:

```
Head_Detection_Grouping_Tracking/
│
├── 0_input_videos/
├── models/
│
├── config.py
├── main.py
├── requirements.txt
├── utils.py
```

## Section 3: Copy the Code

Paste the appropriate content into each file:

* Full configuration block → `config.py`
* Utility functions → `utils.py`
* The full pipeline logic → `main.py`
* Package list → `requirements.txt`

## Section 4: Install Dependencies

Open a terminal and navigate to the project folder:

```
cd path/to/Head_Detection_Grouping_Tracking
```

Install the required packages:

```
pip install -r requirements.txt
```

## Section 5: Run the Pipeline

Prepare the input:

* Add videos inside `0_input_videos/`
* Add your YOLO model (`best.pt`) into `models/`

Execute the main script:

```
python main.py
```

## Section 6: Check Your Output

After completing all stages, an `output` folder will be generated automatically with structured results:

```
Head_Detection_Grouping_Tracking/
│
├── output/
│   ├── 1_dot_videos/
│   │   └── test_video_dots.avi
│   │
│   ├── 2_init_frames/
│   │   ├── test_video_dots_groups.jpg
│   │   └── test_video_dots_groups.json
│   │
│   ├── 3_tracked_videos/
│   │   ├── test_video_dots_tracked.avi
│   │   └── test_video_dots_final_groups.json
│   │
│   ├── 4_validation_logs/
│   │   └── test_video_dots_validation.log
│
├── 0_input_videos/
...
```
