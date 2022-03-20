import csv
from logging import Logger


def reset_logger(logger: Logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilter(f)


def log_csv(csv_file, data):
    with open(csv_file, mode='a', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)
