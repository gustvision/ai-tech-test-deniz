import argparse


def inference_one_example(args):
    pass
    # TODO: Load model

    # TODO: Load and process input file

    # TODO: Prediction

    # TODO: Print the prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    inference_one_example(args)
