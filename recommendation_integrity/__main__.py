"""Small command dispatcher; scientific work stays in the feature modules."""
import argparse
from importlib import import_module
import sys

COMMANDS = {
    'evaluate': '.evaluation.generate',
    'train-sft': '.training.sft',
    'compare': '.evaluation.compare',
    'compare-sft': '.evaluation.matched',
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description='Controlled recommendation-integrity experiments.',
        epilog='Use COMMAND --help for its options. GPU libraries load only when a run starts.',
    )
    parser.add_argument('command', choices=COMMANDS)
    if arguments and arguments[0] in COMMANDS:
        module = import_module(COMMANDS[arguments[0]], package=__package__)
        module.main(arguments[1:])
    else:
        parser.parse_args(arguments)


if __name__ == '__main__':
    main()
