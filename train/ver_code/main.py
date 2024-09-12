import preprocess
import train


def main():
    preprocessed_dataset = preprocess.exec()
    train.exec(preprocessed_dataset)


if __name__ == "__main__":
    main()
