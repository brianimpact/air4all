from data_loader.dataset import AIDataset
import argparse

def main(args):
    AIDataset(taxonomy_name=args.taxonomy_name, dataset_path=args.dataset_path, load_file=True, partitioned=args.partitioned)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxonomy_name', required=True, type=str)
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--partitioned', default=0, type=int)

    args = parser.parse_args()
    args.partitioned = (args.partitioned == 1)
    main(args)
