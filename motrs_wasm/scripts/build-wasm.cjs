const { spawnSync } = require("node:child_process");
const { homedir } = require("node:os");

// Keep personal directory names out of Rust debug information and panic paths.
// Encoded flags preserve paths with spaces and any existing compiler options.
const flags = process.env.CARGO_ENCODED_RUSTFLAGS
    ? process.env.CARGO_ENCODED_RUSTFLAGS.split("\x1f")
    : (process.env.RUSTFLAGS || "").split(/\s+/).filter(Boolean);
flags.push(`--remap-path-prefix=${homedir()}=/local`);

const result = spawnSync(
    "wasm-pack",
    [
        "build",
        "--dev",
        "--target",
        "web",
        "--out-dir",
        "pkg",
        "--out-name",
        "index",
    ],
    {
        stdio: "inherit",
        env: { ...process.env, CARGO_ENCODED_RUSTFLAGS: flags.join("\x1f") },
    }
);

if (result.error) console.error(result.error.message);
process.exit(result.status ?? 1);
