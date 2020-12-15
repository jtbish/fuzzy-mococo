import argparse


def _parse_subspecies_tag(string):
    try:
        return tuple(int(num_mfs) for num_mfs in string.split(","))
    except:
        raise argparse.ArgumentTypeError()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsepcies", type=_parse_subspecies_tag, nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

if __name__ == "__main__":
    main()
