// Generate placeholder PWA icons for Phase 1.
//
// Pure-Node ESM script — no external dependencies. Pixel-encodes the
// PNGs by hand (signature + IHDR + IDAT + IEND) so we don't pull in
// `sharp` / `canvas` / ImageMagick just for placeholder art.
//
// Output:
//   public/icon-192.png            192×192   home-screen icon
//   public/icon-512.png            512×512   PWA splash + Android maskable source
//   public/icon-maskable-512.png   512×512   maskable variant — same glyph,
//                                            sized for the Android "safe area"
//                                            (inner 60% radius from center)
//   public/apple-touch-icon.png    180×180   Safari home-screen icon
//
// Visual: warm-dark `--m-bg` background (#100D0B) with a centered green
// `--m-accent` (#4ADE80) "T" rendered as two rectangles. The "T" is
// pulled in from the canvas edge for the maskable variant so the
// glyph sits inside Android's safe area.
//
// To regenerate after editing this file:
//
//   node scripts/generate-pwa-icons.mjs
//
// Replace the icons whenever final art lands.

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = resolve(__dirname, "..", "public");

const BG = [0x10, 0x0d, 0x0b]; // --m-bg
const FG = [0x4a, 0xde, 0x80]; // --m-accent

// CRC-32 table for PNG chunk checksums.
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = data.length;
  const head = Buffer.alloc(8);
  head.writeUInt32BE(len, 0);
  head.write(type, 4, 4, "ascii");
  const crcInput = Buffer.concat([head.subarray(4, 8), data]);
  const tail = Buffer.alloc(4);
  tail.writeUInt32BE(crc32(crcInput), 0);
  return Buffer.concat([head, data, tail]);
}

function encodeRgbaPng(rgba, width, height) {
  // Insert filter byte (0 = None) at the start of each scanline.
  const stride = width * 4;
  const filtered = Buffer.alloc(height * (stride + 1));
  for (let y = 0; y < height; y++) {
    filtered[y * (stride + 1)] = 0;
    rgba.copy(filtered, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }
  const idat = deflateSync(filtered);

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type: RGBA
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  return Buffer.concat([
    sig,
    chunk("IHDR", ihdr),
    chunk("IDAT", idat),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/**
 * Render a "T" glyph centered on a square warm-dark canvas.
 *
 * Geometry (fractions of canvas size):
 *   horizontalBar:  y in [topMargin, topMargin + barThick],   x in [sideMargin, 1 - sideMargin]
 *   verticalBar:    y in [topMargin,                  bottomY], x in [centerLeft, centerRight]
 *
 * `pad` is the additional inset applied for the maskable variant so
 * the glyph sits inside the Android safe area (≥ 10% all sides).
 */
function paintT(size, { pad = 0 } = {}) {
  const buf = Buffer.alloc(size * size * 4);
  const sideMargin = 0.22 + pad;
  const topMargin = 0.30 + pad;
  const barThick = 0.10;
  const bottomY = 0.78 - pad;
  const centerLeft = 0.46;
  const centerRight = 0.54;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const yp = y / size;
      const xp = x / size;
      const inHorizontal =
        yp >= topMargin &&
        yp <= topMargin + barThick &&
        xp >= sideMargin &&
        xp <= 1 - sideMargin;
      const inVertical =
        yp >= topMargin &&
        yp <= bottomY &&
        xp >= centerLeft &&
        xp <= centerRight;
      const [r, g, b] = inHorizontal || inVertical ? FG : BG;
      const i = (y * size + x) * 4;
      buf[i] = r;
      buf[i + 1] = g;
      buf[i + 2] = b;
      buf[i + 3] = 0xff;
    }
  }
  return encodeRgbaPng(buf, size, size);
}

mkdirSync(PUBLIC_DIR, { recursive: true });

const targets = [
  { name: "icon-192.png", size: 192, opts: {} },
  { name: "icon-512.png", size: 512, opts: {} },
  { name: "icon-maskable-512.png", size: 512, opts: { pad: 0.10 } },
  { name: "apple-touch-icon.png", size: 180, opts: {} },
];

for (const t of targets) {
  const png = paintT(t.size, t.opts);
  const out = resolve(PUBLIC_DIR, t.name);
  writeFileSync(out, png);
  console.log(`wrote ${t.name}  ${t.size}×${t.size}  ${png.length} bytes`);
}
