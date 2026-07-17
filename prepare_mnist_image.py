from pathlib import Path
import argparse

import numpy as np
from PIL import Image


IMG_SIZE = 28


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a MNIST-style PNG/JPEG into a C header."
    )
    parser.add_argument("image", help="Input PNG/JPEG image")
    parser.add_argument(
        "--output",
        default="input_mnist.h",
        help="Output header path",
    )
    parser.add_argument(
        "--invert",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Invert black-on-white images to MNIST white-on-black format",
    )
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}\n"
            "Check the file name and extension. For example, use "
            "'mnist_image.jpg' if your file is a JPEG, or 'mnist_image.png' if it is a PNG."
        )
    
    if not image_path.is_file():
        raise ValueError(f"Input path is not a file: {image_path}")
    
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    pixels = np.asarray(img).astype(np.float32)
    if args.invert == "yes" or (args.invert == "auto" and pixels.mean() > 127.0):
        pixels = 255.0 - pixels

    pixels = np.clip(np.rint(pixels * 127.0 / 255.0), 0, 127).astype(np.int8)
    flat = pixels.reshape(-1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="ascii", newline="\n") as f:
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("const int8_t input_mnist[784] = {\n")
        for row in range(IMG_SIZE):
            start = row * IMG_SIZE
            values = ",".join(str(int(v)) for v in flat[start : start + IMG_SIZE])
            f.write(f"  {values},\n")
        f.write("};\n\n")
        f.write("const unsigned int input_mnist_len = 784;\n")

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
