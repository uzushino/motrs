# Mot16 Challenge

Note: this is just a demo, the script does not evaluate the tracking on MOT16 dataset.

## Usage

```
# Download MOT16 dataset from https://motchallenge.net/data/MOT16.zip .
$ mkdir -p ./examples/mot16_challenge/MOT16
$ unzip MOT16.zip -d ./examples/mot16_challenge/MOT16

# Download font from https://fonts.google.com/specimen/Dela+Gothic+One .
$ mkdir -p ./examples/mot16_challenge/assets/fonts/Dela_Gothic_One/
$ unzip Dela_Gothic_One.zip -d ./examples/mot16_challenge/assets/fonts/Dela_Gothic_One/

$ cargo run --package mot16_challenge
```
