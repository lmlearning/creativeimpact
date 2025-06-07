import json

def setup_aut_objects():
    """
    Defines a list of common objects for the Alternate Uses Task (AUT)
    and saves them to a JSON file.
    """
    aut_objects = [
        "brick",
        "paper-clip",
        "newspaper",
        "coffee mug",
        "shoe",
        "rubber band",
        "car tire",
        "empty bottle",
        "chair",
        "bucket",
        "towel",
        "pencil",
        "book",
        "key",
        "spoon"
    ]

    output_path = "data/aut_objects.json"
    with open(output_path, 'w') as f:
        json.dump({"objects": aut_objects}, f, indent=4)

    print(f"AUT objects saved to {output_path}")

if __name__ == "__main__":
    setup_aut_objects()
