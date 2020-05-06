from src.models import linear_regression, gbm


def main():
    linear_regression.train()
    gbm.train()


if __name__ == '__main__':
    main()
