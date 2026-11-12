import argparse
import os

# Store custom model code alongside the other local model files.
os.environ.setdefault("HF_MODULES_CACHE", ".booksearch/huggingface/modules")

from app import index_book, search

def main():
    # Define CLI, takes in command: index or search, then a pdf or query
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('index').add_argument('txt') # Index method takes a .txt
    search_parser = commands.add_parser('search')
    search_parser.add_argument('query')
    search_parser.add_argument('--verify', action='store_true', help='Check a claim and trim supported passages')
    args = parser.parse_args()
    if args.command == 'index':
        index_book(args.txt)
    else:
        search(args.query, verify=args.verify)
