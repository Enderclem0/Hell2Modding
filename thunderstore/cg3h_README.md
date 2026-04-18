# Hell2ModdingCG3H — temporary fork of Hell2Modding

This is a **temporary** fork of [Hell2Modding](https://thunderstore.io/c/hades-ii/p/Hell2Modding/Hell2Modding/) maintained by **Enderclem** to ship the patches CG3H v3.9.0 depends on, until those patches land in upstream.

**Use upstream Hell2Modding instead once its next release includes these patches.** This fork will be deprecated and delisted then.

## What this fork adds

All changes are framework-level (not CG3H-specific). They're intended for upstream merge.

### New Lua bindings (`rom.data.*`)

- `set_mesh_visible(entry_name, mesh_name, visible)` — flip per-mesh visibility inside an entry by mutating its `GMD+0x4C` (mesh_type) byte. Instant, no draw call, no GPU state change. Main use: per-accessory toggle.
- `swap_to_variant(stock_entry, variant_entry)` — install a hash remap so DoDraw3D reads `variant_entry`'s data when a command queues `stock_entry`. Enables outfit switching without reloading models.
- `restore_stock(stock_entry)` — clear the remap from `swap_to_variant`.
- `populate_entry_textures(entry_name)` — mirror PrepDraw for entries the scene isn't actively walking (variant entries). Writes `GetTexture(+0x40)` into `+0x44` so DoDraw3D's fallback resolves.
- `dump_pool_stats()` — log per-shader-effect vertex-pool cursor + capacity and the global index-buffer cursor + capacity.
- `sanity_check_gmd(entry_name)` — validate the hardcoded `GrannyMeshData` layout on demand. Logs ERRORs if offsets drifted from expected; used by CG3H as a first-frame canary.

### Static pool patches (applied at DLL attach)

Two `mov qword [rsp+X], imm32` byte patches raise the game's static GPU buffer budgets. Both buffers live in the DX12 upload heap (system RAM, not VRAM) so the cost is RAM, not GPU memory.

- **Vertex pool** per shader effect: 64 MB → 128 MB (`sgg::addShaderEffect` 5th arg site).
- **Index pool** global: 32 MB → 64 MB (`sgg::addStaticVertexBuffers` one-shot alloc site).

Without the patches, Hades II's default budget is too tight for mods that add mesh entries — later-loaded meshes (weapons, enemies) fall back to the `Blank_Mesh` placeholder. A plugin-init summary log reports `[OK  ]` / `[SKIP]` per patch so a game update that shifts byte patterns is immediately visible.

### Other

- Shadow-cast code cave (VirtualAlloc-patched JMP at `DoDraw3D + 0x148E4`) so the shadow pass participates in hash remap alongside main/shadow/thumbnail.
- `draw.cpp` module carves all the draw-dispatch hooks into a single file under `lua::hades::draw`.

## Why a fork instead of waiting upstream

Hades II's modding scene moves fast and CG3H v3.9.0 is ready to ship. Keeping users gated on an open-source PR review timeline would delay the release indefinitely. The fork is scoped, narrowly-branded, and will be retired once upstream merges.

## What it does NOT do

- **Not a replacement for upstream Hell2Modding.** If you already have upstream H2M installed for other mods, installing this fork replaces its `d3d12.dll` — whichever is most recent wins. For users of CG3H + non-CG3H mods simultaneously: this fork's DLL is a strict superset of upstream (all upstream APIs still work), so other mods should continue to function. File an issue if not.
- **Not for general use.** If you're not running a CG3H-dependent mod, install upstream Hell2Modding instead.

## Upstream

- Source: <https://github.com/Enderclem0/Hell2Modding/tree/cg3h/v3.9.0>
- Upstream: <https://github.com/SGG-Modding/Hell2Modding>
- CG3H: <https://github.com/Enderclem0/CG3H>

## License

Same license as upstream Hell2Modding. See the repo LICENSE.
