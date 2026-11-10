import argparse

from app import index_book, search

def main():
    # Define CLI, takes in command: index or search, then a pdf or query
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('index').add_argument('txt') # Index method takes a .txt
    commands.add_parser('search').add_argument('query') # Query method takes a str
    args = parser.parse_args()
    if args.command == 'index':
        index_book(args.txt)
    else:
        search(args.query)
