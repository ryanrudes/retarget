/**
 * ANSI → HTML for IPython/Jupyter tracebacks (Shades of Purple terminal palette).
 */
import { AnsiUp } from "https://esm.sh/ansi_up@6.0.6";

const ansiUp = new AnsiUp();
ansiUp.use_classes = false;

const PALETTE_NAMES = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"];

/** @param {[number, number, number][]} rows */
function buildPalette(rows) {
  return rows.map((rgb, index) => ({
    rgb,
    class_name: `ansi-${PALETTE_NAMES[index]}`,
  }));
}

ansiUp.ansi_colors = [
  buildPalette([
    [0x00, 0x00, 0x00],
    [0xec, 0x3a, 0x37],
    [0x3a, 0xd9, 0x00],
    [0xfa, 0xd0, 0x00],
    [0x78, 0x57, 0xfe],
    [0xff, 0x2c, 0x70],
    [0x80, 0xfc, 0xff],
    [0xff, 0xff, 0xff],
  ]),
  buildPalette([
    [0x5c, 0x5c, 0x61],
    [0xec, 0x3a, 0x37],
    [0x3a, 0xd9, 0x00],
    [0xfa, 0xd0, 0x00],
    [0x69, 0x43, 0xff],
    [0xfb, 0x94, 0xff],
    [0x80, 0xfc, 0xff],
    [0xff, 0xff, 0xff],
  ]),
];

/** @param {string} text */
export function hasAnsi(text) {
  return /\u001b\[[0-9;]*[A-Za-z]/.test(text);
}

/** @param {string} text */
export function ansiTextToHtml(text) {
  return ansiUp.ansi_to_html(text);
}
