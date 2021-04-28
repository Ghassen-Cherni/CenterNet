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
    parser.add_argument('--augment_data', default=True, type=bool,
                        help='Use to augment data or not')
    parser.add_argument('--leslie', default=True, type=bool,
                        help='Use Leslie''s experience tu get the best learning rate')
    parser.add_argument('--lr_min', default=1e-8, type=float,
                        help='Minimal learning rate for Leslie''s experience')
    parser.add_argument('--lr_max', default=1, type=float,
                        help='Maximal learning rate for Leslie''s experience')

    args = parser.parse_args()

    return(args)