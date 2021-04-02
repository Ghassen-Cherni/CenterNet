import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run the model")

    parser.add_argument('--epoch', default=20, type=int, help='Number of epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='Value of the learning rate')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size to process the data')
    parser.add_argument('--num_workers', default=4, type=int, help='The total number of workers')
    parser.add_argument('--train_file', default='train.csv', type=str,
                        help='Name of the file to create the dictionnary of the train dataset')
    parser.add_argument('--train_dir', default='train_images/', type=str,
                        help='Name of the directory with the train dataset')
    parser.add_argument('--test_dir', default='test_images/', type=str,
                        help='Name of the directory with the test dataset')

    args = parser.parse_args()

    return(args)