import argparse

from search.multimodal_search import MultimodalSearch
from search.data_processing import data_read
from search.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Image Embeddings CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify the image embedding")
    image_embedding_parser.add_argument("image_path", type=str, help="Image path")

    image_search_parser = subparsers.add_parser("image_search", help="Find documents based on image embeddings")
    image_search_parser.add_argument("image_path", type=str, help="Image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            verify_image_embedding(image_path)

        case "image_search":
            data = data_read("data/movies.json")
            image_path = args.image_path
            multimodal_search = MultimodalSearch(documents=data["movies"])
            results = multimodal_search.search_with_image(image_path)

            for index, element in enumerate(results, start=1):
                print(f"{index}. {element["title"]} (similarity: {float(element['similarity']):.3f})")
                print(f"   {element['description'][:80]}...\n")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()