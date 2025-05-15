# Image Classifier (based on course.fast.ai)

This Python script trains an image classifier using FastAI. It automatically downloads, processes, and verifies images from DuckDuckGo, and then trains a ResNet18 model on them.

It’s meant as a sandbox to explore how dataset size affects training metrics like loss and error rate.

---

## Features

- Automatically downloads and verifies images via DuckDuckGo
- Resizes Images and filters out broken or corrupted ones
- Trains a FastAI ResNet18 classifier
- Logs training metrics (loss, error rate) for different dataset sizes
- Fully configurable through command-line arguments

---

## Requirements

- **Python 3.9+**
- Install dependencies with:

bash
pip install -r requirements.txt

---

## Usage

to run with default values, use:
```
    python3 metric_and_loss_by_data_set_size.py
```

to compare and classify Igunanodon and Toukan Pictures and only use 1 epoch, use:
```
    python3 metric_and_loss_by_data_set_size.py --animals Iguanodon,Toukan --epochs 1```
```

to run 10 times with different seeds:
```
    for i in {0..9}; do
        python3 metric_and_loss_by_data_set_size.py --random_seed $i --metrics_log_file metrics_seed_$i.csv```
    done
```
---

## Plotting

The results of the last Usage example can be plotted by calling the auxliarly plotting.py without Parameters:
```python3 plotting.py```

## Command Line Options

(can also be directly adapted inside the script in the class Config class)
option | description | default
| -------- | -------- | -------- |
--animals | Comma-separated list of classes to search DuckDuckGo for and compare | Duck,Platypus,Hadrosaur,Turtle
--min_photos | Minimum number of images per class | 5
--max_photos | Maximum number of images per class | 45
--step | Step size for the image count | 5
--epochs | Number of training epochs | 20
--validation_set_ratio | Validation split ratio | 0.2
--max_batch_size | Maximum batch size during training | 128
--photo_path | Folder to save downloaded images | beaked animals
--metrics_log_file | Output CSV file for training metrics | metrics.csv
--random_seed | Random seed for reproducibility | 42
--log_level | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | WARNING||

---

## Output Example
metrics.csv:

photo_count | epoch | train_loss | valid_loss | error_rate
| -------- | -------- | -------- |-------- |-------- |
5 | 1 | 2.719990 | 1.397737 | 0.642857
10 | 1 | 3.017617 | 2.907809 | 0.666667
15 | 1 | 2.836110 | 4.097701 | 0.866667
20 | 1 | 3.375271 | 2.754401 | 0.736842
25 | 1 | 3.113768 | 4.236176 | 0.708333
30 | 1 | 3.094889 | 2.367243 | 0.758621
35 | 1 | 3.252741 | 4.101888 | 0.852941
40 | 1 | 2.823795 | 2.030475 | 0.631579

---

## Use Case

- Machine learning experimenting with small image datasets
- Analyzing how training performance changes with different dataset sizes
