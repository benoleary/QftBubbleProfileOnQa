import argparse

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_file")
    parsed_arguments = argument_parser.parse_args()
    # TODO: implement some functionality...
    print(
        "Imagine something wonderful happening, involving input taken from"
        f" {parsed_arguments.input_file}"
    )

if __name__ == '__main__':
    main()
