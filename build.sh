#!/usr/bin/env bash
rm -rf build
meson setup build --buildtype release --prefix $HOME/.local
# meson compile -C build
# meson test -C build
# meson install -C build
