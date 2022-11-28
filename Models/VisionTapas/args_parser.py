import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_parser():
    parser = argparse.ArgumentParser(description='Train VisionTapas Model.')
    # Data Files and Folders
    parser.add_argument('--train-folder', type=str, help='The path to the folder that has the training data')
    parser.add_argument('--validation-folder', type=str, help='The path to the folder that has the validation data')
    parser.add_argument('--qa-train-file', type=str, help='The path to the file that has the Q/A pairs for the training data')
    parser.add_argument('--qa-val-file', type=str, help='The path to the file that has the Q/A pairs for the validation data')
    parser.add_argument('--out-dir', type=str, help='The output directory where the checkpoints will be saved')

    # Training Parameters
    parser.add_argument('--EPOCHS', type=int, default=4, help='Number of Epochs')
    parser.add_argument('--batch-size', type=int, default = 8, help='Training batch size')
    parser.add_argument('--eval-batch-size', type=int, default = 8, help='Validation batch size')
    parser.add_argument('--eval-num', type=int, default = 5000, help='Number of random validation examples you want to evaluate the model on at the end of each eval_steps')
    parser.add_argument('--eval-steps', type=int, default = 5000, help='The number of steps between each two consecutive evaluations during the training')
    parser.add_argument('--learning-rate', type=float, default = 1e-05, help='learning rate')
    parser.add_argument('--warmup-ratio', type=float, default = 0.1, help='warmup ratio')
    parser.add_argument('--weight-decay', type=float, default = 0.01, help='weight decay')
    parser.add_argument('--num-workers', type=int, default = 8, help='number of workers')

    parser.add_argument('--num-labels', type=int, default=55, help='The number of output classification labels')
    parser.add_argument('--fixed-vocab-file', type=str, help='Fixed Vocab File for Predictions')
    parser.add_argument('--checkpoint-folder', type=str, default=None, help='checkpoint_folder')


    return parser


def get_parserQA():
    parser = argparse.ArgumentParser(description='Train VisionTapas Model.')
    # Data Files and Folders
    parser.add_argument('--train-folder', type=str, help='The path to the folder that has the training data')
    parser.add_argument('--validation-folder', type=str, help='The path to the folder that has the validation data')
    parser.add_argument('--qa-train-file', type=str, help='The path to the file that has the Q/A pairs for the training data')
    parser.add_argument('--qa-val-file', type=str, help='The path to the file that has the Q/A pairs for the validation data')
    parser.add_argument('--out-dir', type=str, help='The output directory where the checkpoints will be saved')

    # Training Parameters
    parser.add_argument('--EPOCHS', type=int, default=4, help='Number of Epochs')
    parser.add_argument('--batch-size', type=int, default = 8, help='Training batch size')
    parser.add_argument('--eval-batch-size', type=int, default = 8, help='Validation batch size')
    parser.add_argument('--eval-num', type=int, default = 5000, help='Number of random validation examples you want to evaluate the model on at the end of each eval_steps')
    parser.add_argument('--eval-steps', type=int, default = 5000, help='The number of steps between each two consecutive evaluations during the training')
    parser.add_argument('--learning-rate', type=float, default = 1e-05, help='learning rate')
    parser.add_argument('--warmup-ratio', type=float, default = 0.1, help='warmup ratio')
    parser.add_argument('--weight-decay', type=float, default = 0.01, help='weight decay')
    parser.add_argument('--num-workers', type=int, default = 8, help='number of workers')

    parser.add_argument('--num-labels', type=int, default=55, help='The number of output classification labels')
    parser.add_argument('--fixed-vocab-file', type=str, help='Fixed Vocab File for Predictions')
    parser.add_argument('--checkpoint-folder', type=str, default=None, help='checkpoint_folder')

    # TaPas Paramaters
    parser.add_argument('--answer-loss-cutoff', type=int, default=50, help='Cutoff for answer loss computation')
    parser.add_argument('--cell-selection-preference', type=float, default=0.001, help='Cell Selection Preference Score')
    parser.add_argument('--select-one-column', type=str2bool, default=True, help='Select One Column or all when computing TaPas Loss')

    return parser


def get_parserInference():
    parser = argparse.ArgumentParser(description='Inference VisionTapas Model.')
    # Data Files and Folders
    parser.add_argument('--validation-folder', type=str, help='The path to the folder that has the validation data')
    parser.add_argument('--qa-val-file', type=str, help='The path to the file that has the Q/A pairs for the validation data')
    parser.add_argument('--out-dir', type=str, help='The output directory where the checkpoints will be saved')

    # Training Parameters
    parser.add_argument('--eval-batch-size', type=int, default = 8, help='Validation batch size')
    parser.add_argument('--num-workers', type=int, default = 8, help='number of workers')

    parser.add_argument('--fixed-vocab-file', type=str, help='Fixed Vocab File for Predictions')
    parser.add_argument('--checkpoint-folder', type=str, default=None, help='checkpoint_folder')

    # TaPas Paramaters
    parser.add_argument('--answer-loss-cutoff', type=int, default=50, help='Cutoff for answer loss computation')
    parser.add_argument('--cell-selection-preference', type=float, default=0.001, help='Cell Selection Preference Score')
    parser.add_argument('--select-one-column', type=str2bool, default=True, help='Select One Column or all when computing TaPas Loss')

    return parser