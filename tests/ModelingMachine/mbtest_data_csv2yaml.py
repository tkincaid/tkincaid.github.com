#!/usr/bin/python
"""Usage: mbtest_data_csv2yaml [-h] <in_file> [--out=<out_file>]

Convert mbtest_data files in semi-colon separated CSV format to YAML format.

Arguments:
  --out=<out_file>  The output file; if - use stdout [default: -].

Options:
  -h --help
"""
import sys
import csv
import yaml
from docopt import docopt


def file_handle(fname, mode='r'):
    if fname.strip() == '-':
        return sys.stdout
    else:
        return open(fname, mode)


FIELDS = ['dataset_name', 'target', 'metric', 'worker_size']


def mbtest_data_csv_to_yaml_format(csv_reader):
    out = []
    for row in csv_reader:
        row = dict(zip(FIELDS, row))
        out.append(row)
    return out


def main(args):
    in_fd = file_handle(args['<in_file>'])
    out_fd = file_handle(args['--out'], mode='w+')
    csv_reader = csv.reader(in_fd, delimiter=';')
    try:
        out = mbtest_data_csv_to_yaml_format(csv_reader)
        yaml.dump(out, out_fd, default_flow_style=False)
    finally:
        in_fd.close()
        out_fd.close()


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
