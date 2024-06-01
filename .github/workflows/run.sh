#!/bin/sh
set -eux
./mnist.sh
cargo test -r -- --nocapture

