import argparse
import os
import sys
from mimetypes import guess_type

from llm.gemini_client import multimodal_llm_query


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_subparsers(dest="command", help="Available commands")
    parser.add_argument("--image", type=str, help="Image path")
    parser.add_argument("--query", type=str, help="Text query to rewrite")
    args = parser.parse_args()

    match args.command:
        case _:
            image = args.image
            query = args.query

            mime, _ = guess_type(image)
            mime = mime or "image/jpeg"

            if not os.path.exists(image):
                print(f"Image {image} not found")
                sys.exit(1)

            with open(image, "rb") as file:
                image_content = file.read()
                result = multimodal_llm_query(query, image_content, mime)

                print(f"Rewritten query: {result[0]}")
                if result[1] is not None:
                    print(f"Total tokens:       {result[1].total_token_count}")

if __name__ == "__main__":
    main()