#!/bin/sh
set -eux
./mnist.sh
./timemachine.sh
cargo test -r -- --nocapture

