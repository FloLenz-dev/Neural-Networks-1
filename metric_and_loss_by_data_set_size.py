import socket
import time
from pathlib import Path
from typing import List, Dict, Union
import logging
import argparse
from dataclasses import dataclass

from tqdm import tqdm
import pandas as pd
from duckduckgo_search import DDGS
from fastai.vision.all import (
    get_image_files,
    resize_images,
    verify_images,
    download_images,
    DataBlock,
    DataLoaders,
    ImageBlock,
    CategoryBlock,
    RandomSplitter,
    parent_label,
    Resize,
    vision_learner,
    Learner,
    resnet18,
    error_rate,
)
from fastcore.all import L

logger = logging.getLogger(__name__)

INTERNET_TEST_HOST = "1.1.1.1"  # Cloudflare DNS
INTERNET_TEST_PORT = 53  # Cloudflare DNS

SOCKET_TIMEOUT = 1

RATE_LIMIT_TIMEOUT_SECONDS = 5

INPUT_MAXIMUM_PICTURE_SIZE = 400
TRAINING_MAXIMUM_PICTURE_SIZE = 192


@dataclass
class Config:
    """
    Holds default configuration for image acquisition and training.
    All values can be overridden via command-line arguments.
    """

    animals: str = "Duck,Platypus,Hadrosaur,Turtle"
    min_photos: int = 5
    max_photos: int = 45
    step: int = 5
    epochs: int = 20
    validation_set_ratio: float = 0.2
    max_batch_size: int = 128
    photo_path: str = "photos"
    metrics_log_file: str = "metrics.csv"
    random_seed: int = 42
    input_max_pic_size: int = 400
    training_max_pic_size: int = 192
    log_level: str = "WARNING"


def setup_logging(level_str: str = "WARNING") -> None:
    """Configure the logging system."""

    level = getattr(logging, level_str.upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
    )

def parse_args(default: Config)  -> argparse.Namespace:
    """parse the command line inputs"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a", "--animals",
        type=str,
        default=default.animals,
        help="comma seperated list of classes (Default: Duck,Platypus,Hadrosaur,Turtle)"
    )

    parser.add_argument(
        "-m", "--min_photos", type=int, default=default.min_photos,
        help="minimal number of photos to try (Default: 5)"
    )

    parser.add_argument(
        "-M", "--max_photos", type=int, default=default.max_photos,
        help="maximum number of photos to try (Default: 45)"
    )

    parser.add_argument(
        "-s", "--step", type=int, default=default.step,
        help="step size for photo count (Default: 5)"
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=default.epochs,
        help="number of training epochs (Default: 20)"
    )

    parser.add_argument(
        "-v", "--validation_set_ratio", type=float, default=default.validation_set_ratio,
        help="share of validation set (Default: 0.2)"
    )

    parser.add_argument(
        "-b", "--max_batch_size", type=int, default=default.max_batch_size,
        help="maximum of training batch size (Default: 128)"
    )

    parser.add_argument(
        "-p", "--photo_path", type=str, default=default.photo_path,
        help="path to save picture (Default: photos)"
    )

    parser.add_argument(
        "-f", "--metrics_log_file", type=str, default=default.metrics_log_file,
        help="name of metrics log file (Default: metrics.csv)"
    )

    parser.add_argument(
        "-r", "--random_seed", type=int, default=default.random_seed,
        help="value of seed for pseudo randomness (Default: 42)"
    )
    parser.add_argument(
    "--log_level", type=str, default=default.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set logging level (Default: WARNING)"
    )
    return parser.parse_args()

def check_internet_connection() -> None:
    """Check for an active internet connection."""

    try:
        socket.setdefaulttimeout(SOCKET_TIMEOUT)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            (INTERNET_TEST_HOST, INTERNET_TEST_PORT)
        )
        logger.info("Internet connection verified.")
    except socket.error:
        logger.error("No Internet connection found. Please check and try again. Exiting for now")
        raise ConnectionError("No Internet connection found. . Please check and try again. Exiting for now")


def fetch_image_urls(keywords: str, max_images: int) -> List[str]:
    """Search for image URLs using DuckDuckGo."""

    return L(DDGS().images(keywords, max_results=max_images)).itemgot("image")


def prepare_images(animal: str, photo_path: Path, photo_count: int) -> None:
    """Download, resize, and verify images for a given animal."""

    animal_path = determine_path(photo_count, photo_path, animal)

    #check whether downloads have already been done succesfully
    if animal_path.exists() and len(get_image_files(animal_path)) > 0: 
        logger.warning(f"{animal}: download path already exists with {len(get_image_files(animal_path))}, skip download")
        return

    animal_path.mkdir(exist_ok=True, parents=True)

    try:
        image_urls = fetch_image_urls(f"{animal} photo", photo_count) #use DuckDuckGo to fetch URLS of photos
        download_images(animal_path, urls=image_urls) #use download the fetched photos
    except Exception as e:
        logger.warning(f"Image download failed for {animal}: {e}")
        return

    time.sleep(RATE_LIMIT_TIMEOUT_SECONDS)  # avoid rate limiting

    resize_images(animal_path, max_size=INPUT_MAXIMUM_PICTURE_SIZE, dest=animal_path) # resize large photos to save time in training
    failed = verify_images(get_image_files(animal_path))
    failed.map(Path.unlink)
    if (len(failed) > 0):
        logger.info(f"{animal}: {len(failed)} failed images removed")
    return


def create_dataloaders(photo_count_path: Path, validation_set_ratio: float, random_seed: int, max_batch_size: int) -> DataLoaders:
    """Create FastAI dataloaders from image data."""

    # determine suitable batch size (speed up training, but make sure not have less photos than batch size, which would yield a hard fastai crash
    training_set_share = 1 - validation_set_ratio
    training_set_size = int(len(get_image_files(photo_count_path)) * training_set_share)
    batch_size = min(training_set_size, max_batch_size)

    #prepare the data loader
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=validation_set_ratio, seed=random_seed),
        get_y=parent_label,
        item_tfms=[Resize(TRAINING_MAXIMUM_PICTURE_SIZE, method="squish")],
    ).dataloaders(photo_count_path, bs=batch_size)


def train_model(dataloaders: DataLoaders, epochs: int) -> Learner:
    """Train a ResNet18 model on the given dataloaders."""

    learner = vision_learner(dataloaders, resnet18, metrics=error_rate)
    learner.fine_tune(epochs)
    return learner


def collect_metrics(learner: Learner, photo_count: int ) -> List[Dict[str, Union[float, int]]]:
    """Extract training metrics from the learner."""

    metrics = []
    for i, (train_loss, valid_loss, error) in enumerate(learner.recorder.values):
        metrics.append(
            {
                "photo_count": photo_count,
                "epoch": i + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "error_rate": error,
            }
        )
    return metrics


def determine_path(photo_count: int, download_path: Path, animal: str = "") -> Path:
    """Construct a path instance for storing images based on animal name and photo count."""
    return download_path / str(photo_count) / animal

def save_metrics_to_file(all_metrics: List[Dict[str, Union[float, int]]], metrics_log_file: str) -> None:
    """Save metrics data to CSV file"""

    log_df = pd.DataFrame(all_metrics)
    log_df.to_csv(metrics_log_file, index=False)
    logger.info("Log saved to CSV.")
    
def main() -> None:
    """Run the full training pipeline (check connection, download photo, train model, save metrics) for different photo counts."""
    
    default_config = Config()
    args = parse_args(default_config)
    photo_path = Path(args.photo_path)
    all_metrics = []

    setup_logging(args.log_level)

    check_internet_connection()

    for photo_count in tqdm(range(args.min_photos,args.max_photos,args.step), desc="Photo count loop"):
        photo_count_path = determine_path(photo_count, photo_path)

        # get animal photos
        for animal in args.animals.split(","):
            prepare_images(animal, photo_path, photo_count)

        # create and train the model
        try:
            dataloaders = create_dataloaders(photo_count_path, args.validation_set_ratio, args.random_seed, args.max_batch_size)
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Dataloaders für {photo_count}: {e}")
            continue
        learner = train_model(dataloaders, args.epochs)

	#collect metrics
        metrics = collect_metrics(learner, photo_count)
        all_metrics.extend(metrics)

    # Save metrics to CSV
    save_metrics_to_file(all_metrics, args.metrics_log_file) 


if __name__ == "__main__":
    main()
