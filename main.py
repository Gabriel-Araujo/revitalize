import cv2
from utils import get_args


def main():
    config = get_args()
    print(config)

    image = cv2.imread(config.file_path)

    if image is None:
        print("Error: Image not found")
        exit("Image not found")


if __name__ == "__main__":
    main()
