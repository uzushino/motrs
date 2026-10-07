# MOT16 Challenge

This demo displays tracking results; it does not evaluate the tracking on the MOT16 dataset.

Run these commands from the repository root with the Rust toolchain selected by
`rust-toolchain.toml`:

```sh
# Download MOT16.zip from https://motchallenge.net/data/MOT16.zip.
unzip MOT16.zip -d motrs/examples/mot16_challenge
cargo run --locked -p mot16_challenge
```

The viewer reads `motrs/examples/mot16_challenge/MOT16/train/MOT16-04/gt/gt.txt`
and the images in the sequence's `img1` directory. Frame files use six digits,
for example `000001.jpg`.

Optionally download [Dela Gothic One](https://fonts.google.com/specimen/Dela+Gothic+One)
and place `DelaGothicOne-Regular.ttf` in
`motrs/examples/mot16_challenge/assets/fonts/Dela_Gothic_One/` to display track IDs.
The viewer falls back to Arial on macOS/Windows or DejaVu Sans on Linux.
If no supported font is available, it still draws tracking and detection rectangles.
The dataset and font are not required to build or run the tests.
You can select a different sequence by passing its directory:

```sh
cargo run --locked -p mot16_challenge -- /path/to/MOT16/train/MOT16-02
```

This viewer uses annotation boxes as observations, without adding random noise
or copying the annotated object IDs. Empty frames are preserved.


To validate every annotated frame without opening a window, run:

```sh
cargo run --locked -p mot16_challenge -- --check
# Or select another sequence directory:
cargo run --locked -p mot16_challenge -- --check /path/to/MOT16/train/MOT16-04
```

This mode loads every image and performs the same tracking and overlay drawing
as the GUI. It reports input/rendering failures; it does not calculate MOT metrics.
