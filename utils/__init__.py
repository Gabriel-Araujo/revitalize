from custom_types import Config
import sys

__all__ = ["get_args"]


def get_args():
    if sys.argv.__len__() < 2:
        print("usage: python3 main.py <filetype> <filepath> debug")
        exit("Missing params")

    args = sys.argv
    _ = args.pop(0)

    if "document" in args:
        file_type = "document"
    elif "photo" in args:
        file_type = "photo"
    else:
        print("usage: python3 main.py <filetype> <filepath> debug")
        exit("No file type specified")

    args.remove(file_type)

    if "debug" in args:
        debug_mode = True
        args.remove("debug")
    else:
        debug_mode = False

    if args.__len__() > 1:
        print("usage: python3 main.py <filetype> <filepath> debug")
        exit("Too many arguments")

    if args.__len__() < 1:
        print("usage: python3 main.py <filetype> <filepath> debug")
        exit("Missing filepath")

    return Config(file_type, args.pop(0), debug_mode)
