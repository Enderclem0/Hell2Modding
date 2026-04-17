/// @file draw.cpp
/// @brief Draw-call control for 3D model entries.
///
/// Provides two Lua APIs for controlling 3D model rendering at runtime:
///
///   rom.data.set_draw_visible(entry_name, visible)
///     Suppress or restore draw calls for a model entry by HashGuid.
///
///   rom.data.set_draw_remap(source_entry, target_entry)
///   rom.data.clear_draw_remap(source_entry)
///     Redirect a model entry to draw a different mModelData entry.
///     Enables outfit/variant switching without reload.
///
///   rom.data.add_model_entry(entry_name)
///     Register a custom entry name to be loaded during startup.
///     Must be called during plugin init, before model loading.
///
/// Three draw functions (DoDraw3D, DoDrawShadow3D, DoDraw3DThumbnail)
/// share the PDB signature `static void(const vector<RenderMesh*>&,
/// uint, int, HashGuid)` and are handled with standard detour hooks.
///
/// DoDrawShadowCast3D has a different signature (no HashGuid param).
/// A manual code cave patched into the draw dispatch handles both
/// visibility and remapping for all four draw functions by intercepting
/// the draw entry's hash at [r10+0x28] before the variant branch.

#include "draw.hpp"

#include "hades_ida.hpp"

#include <hades2/pdb_symbol_map.hpp>
#include <hooks/hooking.hpp>
#include <lua/lua_manager.hpp>
#include <atomic>
#include <map>
#include <memory/gm_address.hpp>
#include <mutex>
#include <shared_mutex>
#include <string/string.hpp>
#include <tuple>
#include <unordered_map>

namespace lua::hades::draw
{
	// ─── State ─────────────────────────────────────────────────────────

	static std::shared_mutex g_mutex;
	static std::unordered_set<unsigned int> g_hidden_entries;
	static std::unordered_map<unsigned int, unsigned int> g_remap; // original → variant

	// Fast-path flag for the code cave — avoids the function-call overhead
	// on every draw entry when nothing is active (hidden or remapped).
	static volatile uint8_t g_any_active = 0;

	static void update_active_flag()
	{
		g_any_active = (g_hidden_entries.empty() && g_remap.empty()) ? 0 : 1;
	}

	// Called from the code cave via function pointer.
	// Returns: 0 = pass through, 1 = hidden (skip), 2 = remapped (*out_hash set)
	static int check_draw_entry(uint32_t hash, uint32_t* out_hash)
	{
		std::shared_lock l(g_mutex);
		auto it = g_remap.find(hash);
		if (it != g_remap.end())
		{
			*out_hash = it->second;
			return 2;
		}
		if (g_hidden_entries.count(hash))
			return 1;
		return 0;
	}

	// Lua API: Function
	// Table: data
	// Name: load_model_entry
	// Param: entry_name: string: Entry name to load into mModelData.
	// Calls sgg::Granny3D::LoadModelData directly.  The entry's GPK must
	// already be registered via `add_granny_file`.
	//
	// IMPORTANT: Must be called AFTER LoadAllModelAndAnimationData completes
	// (e.g. from rom.on_import.post("Main.lua")) but BEFORE the first frame.
	// Calling during plugin init corrupts weapon models; calling mid-session
	// crashes App::UpdateAndDraw.
	//
	// **Example Usage:**
	// ```lua
	// rom.on_import.post(function(script_name)
	//     if script_name == "Main.lua" then
	//         rom.data.add_granny_file("MyVariant.gpk", path)
	//         rom.data.load_model_entry("MyVariant")
	//     end
	// end)
	// ```
	// Intern a string into the game's HashGuid system.
	// Unlike Lookup (which only finds existing strings), StringIntern
	// CREATES a hash entry for new strings.  Required for custom variant
	// names that the game has never seen.
	static sgg::HashGuid intern_string(const std::string& name)
	{
		static auto StringIntern = big::hades2_symbol_to_address["sgg::HashGuid::StringIntern"]
		    .as_func<uint32_t(const char*, int)>();

		sgg::HashGuid guid{};
		if (StringIntern)
			guid.mId = StringIntern(name.c_str(), static_cast<int>(name.size()));
		return guid;
	}

	static void load_model_entry(const std::string& entry_name)
	{
		static auto LoadModelData_fn = big::hades2_symbol_to_address["sgg::Granny3D::LoadModelData"]
		    .as_func<void(sgg::HashGuid)>();

		if (!LoadModelData_fn)
		{
			LOG(ERROR) << "draw: LoadModelData symbol not found";
			return;
		}

		sgg::HashGuid guid = intern_string(entry_name);
		if (guid.mId == 0)
		{
			LOG(WARNING) << "draw: intern returned 0 for '" << entry_name << "'";
			return;
		}

		LoadModelData_fn(guid);
		LOG(INFO) << "draw: loaded " << entry_name << " (hash=" << guid.mId << ")";
	}

	// ─── Detour hooks (DoDraw3D, DoDrawShadow3D, DoDraw3DThumbnail) ──

	// Helper: apply remap + hidden check on a hash.  Returns true if the
	// entry should be drawn (possibly with a modified hash).
	static bool apply_draw_filter(sgg::HashGuid& hash)
	{
		std::shared_lock l(g_mutex);
		auto it = g_remap.find(hash.mId);
		if (it != g_remap.end())
			hash.mId = it->second;
		return g_hidden_entries.count(hash.mId) == 0;
	}

	static void hook_DoDraw3D(void* vec_ref, unsigned int index, int param, sgg::HashGuid hash)
	{
		{
			std::shared_lock l(g_mutex);
			auto it = g_remap.find(hash.mId);
			if (it != g_remap.end())
				hash.mId = it->second;
			if (g_hidden_entries.count(hash.mId))
				return;
		}
		big::g_hooking->get_original<hook_DoDraw3D>()(vec_ref, index, param, hash);
	}

	// Shadow + thumbnail paths honor the HIDDEN set (v3.8 visibility gate) but
	// do NOT participate in hash remap.  Reasoning: DoDrawShadowCast3D is
	// remapped via a manual code cave that requires hardcoded byte offsets;
	// when those don't match (post-game-update) the cave fails open and
	// shadow cast renders with the STOCK hash.  Remapping the other shadow
	// paths in that state leaves the engine with inconsistent per-entry
	// state (main=variant, shadow_cast=stock, shadow3D=variant) which
	// appears to wedge the render thread.  Until the cave is re-fitted on
	// every patch, shadow/thumbnail follow stock.  The main DoDraw3D (body)
	// is still remapped — that's where the visual swap happens.
	static void hook_DoDrawShadow3D(void* vec_ref, unsigned int index, int param, sgg::HashGuid hash)
	{
		{
			std::shared_lock l(g_mutex);
			if (g_hidden_entries.count(hash.mId))
				return;
		}
		big::g_hooking->get_original<hook_DoDrawShadow3D>()(vec_ref, index, param, hash);
	}

	static void hook_DoDraw3DThumbnail(void* vec_ref, unsigned int index, int param, sgg::HashGuid hash)
	{
		{
			std::shared_lock l(g_mutex);
			if (g_hidden_entries.count(hash.mId))
				return;
		}
		big::g_hooking->get_original<hook_DoDraw3DThumbnail>()(vec_ref, index, param, hash);
	}

	// ─── Code cave for DoDrawShadowCast3D ─────────────────────────────
	//
	// Overwrites 7 bytes at the dispatch's shadow-flag check:
	//   cmp byte [r10+0x2d], 0   (5B)
	//   je  <main_path>          (2B)
	// with:
	//   jmp code_cave            (5B)
	//   nop; nop                 (2B)
	//
	// The cave checks [r10+0x28] against the hidden set.  If hidden it
	// skips to the loop-next target; otherwise it replays the original
	// cmp+je and resumes normal execution.

	static void install_shadow_cast_patch(uintptr_t doDraw3D_addr)
	{
		// All offsets are relative to the DoDraw3D PDB symbol address.
		uintptr_t patch_site      = doDraw3D_addr + 0x148E4; // cmp byte [r10+0x2d], 0
		uintptr_t shadow_continue = doDraw3D_addr + 0x148EB; // shadow param setup
		uintptr_t main_continue   = doDraw3D_addr + 0x148FF; // hash load + main path
		uintptr_t loop_next       = doDraw3D_addr + 0x14AC1; // inc rsi (loop next)

		// Verify expected bytes before patching.
		const uint8_t expected[] = {0x41, 0x80, 0x7A, 0x2D, 0x00, 0x74, 0x14};
		if (memcmp((void*)patch_site, expected, 7) != 0)
		{
			LOG(ERROR) << "draw: shadow patch byte mismatch at "
			           << HEX_TO_UPPER(patch_site) << " — skipping";
			return;
		}

		// Allocate code cave within ±2GB of the patch site (required for rel32 jmp).
		void* cave = nullptr;
		uintptr_t try_addr = patch_site & ~0xFFFFULL;
		for (uintptr_t offset = 0x10000; offset < 0x7FFF0000ULL; offset += 0x10000)
		{
			cave = VirtualAlloc((void*)(try_addr - offset), 4096,
			                    MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
			if (cave) break;
			cave = VirtualAlloc((void*)(try_addr + offset), 4096,
			                    MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
			if (cave) break;
		}
		if (!cave)
		{
			LOG(ERROR) << "draw: VirtualAlloc failed for shadow code cave";
			return;
		}

		uintptr_t cave_addr = (uintptr_t)cave;
		uintptr_t data_base = cave_addr + 0x80; // data section after code

		// Data section (5 pointers, filled at runtime)
		*(uintptr_t*)(data_base + 0x00) = (uintptr_t)&g_any_active;
		*(uintptr_t*)(data_base + 0x08) = (uintptr_t)&check_draw_entry;
		*(uintptr_t*)(data_base + 0x10) = shadow_continue;
		*(uintptr_t*)(data_base + 0x18) = main_continue;
		*(uintptr_t*)(data_base + 0x20) = loop_next;

		// Code section — hand-assembled x86-64
		uint8_t* p = (uint8_t*)cave;
		auto emit = [&](std::initializer_list<uint8_t> bytes) {
			for (auto b : bytes) *p++ = b;
		};
		auto emit_rel32 = [&](int32_t disp) {
			memcpy(p, &disp, 4); p += 4;
		};
		auto cur = [&]() -> size_t { return (size_t)(p - (uint8_t*)cave); };
		auto rip_data = [&](size_t data_off, size_t rest) -> int32_t {
			return (int32_t)((data_base + data_off) - ((uintptr_t)p + rest));
		};

		// Fast path: if nothing is active (no hidden, no remap), skip to original
		emit({0x48, 0x8B, 0x05}); emit_rel32(rip_data(0x00, 4)); // mov rax, [rip+g_any_active]
		emit({0x80, 0x38, 0x00});                                  // cmp byte [rax], 0
		emit({0x74}); size_t je_fast = cur(); emit({0x00});         // je .not_hidden

		// Slow path: call check_draw_entry(ecx=hash, rdx=&out_hash)
		// Returns: eax=0 pass, eax=1 hidden, eax=2 remapped
		emit({0x51, 0x52});                                         // push rcx; push rdx
		emit({0x48, 0x83, 0xEC, 0x30});                             // sub rsp, 0x30 (shadow + local)
		emit({0x41, 0x8B, 0x4A, 0x28});                             // mov ecx, [r10+0x28]
		emit({0x48, 0x8D, 0x54, 0x24, 0x28});                      // lea rdx, [rsp+0x28] (out_hash)
		emit({0x48, 0x8B, 0x05}); emit_rel32(rip_data(0x08, 4));   // mov rax, [rip+check_draw_entry]
		emit({0xFF, 0xD0});                                         // call rax
		emit({0x83, 0xF8, 0x01});                                   // cmp eax, 1
		emit({0x74}); size_t je_skip = cur(); emit({0x00});          // je .skip
		emit({0x83, 0xF8, 0x02});                                   // cmp eax, 2
		emit({0x74}); size_t je_remap = cur(); emit({0x00});         // je .remap
		// eax=0: pass through
		emit({0x48, 0x83, 0xC4, 0x30});                             // add rsp, 0x30
		emit({0x5A, 0x59});                                         // pop rdx; pop rcx
		emit({0xEB}); size_t jmp_not_hidden = cur(); emit({0x00});   // jmp .not_hidden

		// .remap — rewrite [r10+0x28] with remapped hash, then pass through
		size_t remap_off = cur();
		*((uint8_t*)cave + je_remap) = (uint8_t)(remap_off - je_remap - 1);
		emit({0x8B, 0x44, 0x24, 0x28});                             // mov eax, [rsp+0x28] (out_hash)
		emit({0x41, 0x89, 0x42, 0x28});                             // mov [r10+0x28], eax (overwrite!)
		emit({0x48, 0x83, 0xC4, 0x30});                             // add rsp, 0x30
		emit({0x5A, 0x59});                                         // pop rdx; pop rcx
		emit({0xEB}); size_t jmp_not_hidden2 = cur(); emit({0x00});  // jmp .not_hidden

		// .not_hidden — replay original instructions
		size_t not_hidden = cur();
		*((uint8_t*)cave + je_fast) = (uint8_t)(not_hidden - je_fast - 1);
		*((uint8_t*)cave + jmp_not_hidden) = (uint8_t)(not_hidden - jmp_not_hidden - 1);
		*((uint8_t*)cave + jmp_not_hidden2) = (uint8_t)(not_hidden - jmp_not_hidden2 - 1);
		emit({0x41, 0x80, 0x7A, 0x2D, 0x00});                      // cmp byte [r10+0x2d], 0
		emit({0x74}); size_t je_main = cur(); emit({0x00});          // je .main
		emit({0xFF, 0x25}); emit_rel32(rip_data(0x10, 4));          // jmp [shadow_continue]

		// .main — non-shadow path
		size_t main_off = cur();
		*((uint8_t*)cave + je_main) = (uint8_t)(main_off - je_main - 1);
		emit({0xFF, 0x25}); emit_rel32(rip_data(0x18, 4));          // jmp [main_continue]

		// .skip — entry is hidden, advance loop
		size_t skip_off = cur();
		*((uint8_t*)cave + je_skip) = (uint8_t)(skip_off - je_skip - 1);
		emit({0x48, 0x83, 0xC4, 0x30});                             // add rsp, 0x30
		emit({0x5A, 0x59});                                         // pop rdx; pop rcx
		emit({0xFF, 0x25}); emit_rel32(rip_data(0x20, 4));          // jmp [loop_next]

		LOG(INFO) << "draw: shadow cave at " << HEX_TO_UPPER(cave_addr)
		          << " (" << cur() << " bytes)";

		// Overwrite the original 7 bytes with jmp + nops.
		DWORD old_protect;
		VirtualProtect((void*)patch_site, 7, PAGE_EXECUTE_READWRITE, &old_protect);
		int32_t rel = (int32_t)(cave_addr - (patch_site + 5));
		uint8_t patch[] = {0xE9, 0, 0, 0, 0, 0x90, 0x90};
		memcpy(patch + 1, &rel, 4);
		memcpy((void*)patch_site, patch, 7);
		VirtualProtect((void*)patch_site, 7, old_protect, &old_protect);
		FlushInstructionCache(GetCurrentProcess(), (void*)patch_site, 7);

		LOG(INFO) << "draw: shadow patch installed at " << HEX_TO_UPPER(patch_site);
	}

	// ─── Lua binding ──────────────────────────────────────────────────

	// Lua API: Function
	// Table: data
	// Name: set_draw_visible
	// Param: entry_name: string: Model entry name (e.g. "HecateBattle_Mesh").
	// Param: visible: boolean: true to show, false to hide.
	// Toggles visibility of a model entry by suppressing its draw calls
	// (DoDraw3D, DoDrawShadow3D, DoDraw3DThumbnail, DoDrawShadowCast3D).
	// Takes effect immediately — no rebuild, no restart, no data mutation.
	//
	// **Example Usage:**
	// ```lua
	// rom.data.set_draw_visible("HecateBattle_Mesh", false)
	// ```
	static void set_draw_visible(const std::string& entry_name, bool visible)
	{
		static auto Lookup = *big::hades2_symbol_to_address["sgg::HashGuid::Lookup"]
		    .as_func<sgg::HashGuid*(sgg::HashGuid*, const char*, size_t)>();

		sgg::HashGuid guid{};
		Lookup(&guid, entry_name.c_str(), entry_name.size());

		if (guid.mId == 0)
		{
			LOG(WARNING) << "draw: hash=0 for '" << entry_name
			             << "' — hash system not ready, skipping";
			return;
		}

		size_t old_size, new_size;
		{
			std::unique_lock l(g_mutex);
			old_size = g_hidden_entries.size();
			if (visible)
				g_hidden_entries.erase(guid.mId);
			else
				g_hidden_entries.insert(guid.mId);
			new_size = g_hidden_entries.size();
		}

		update_active_flag();

		LOG(INFO) << "draw: " << entry_name
		          << (visible ? " SHOW" : " HIDE")
		          << " (hash=" << guid.mId
		          << ", set " << old_size << " -> " << new_size << ")";
	}

	// Lua API: Function
	// Table: data
	// Name: set_draw_remap
	// Param: source_entry: string: Original model entry name (e.g. "HecateBattle_Mesh").
	// Param: target_entry: string: Variant entry name to draw instead.
	// Redirects a model entry to draw a different mModelData entry.
	// Takes effect immediately.  Use `clear_draw_remap` to revert.
	//
	// **Example Usage:**
	// ```lua
	// rom.data.set_draw_remap("HecateBattle_Mesh", "HecateBattle_Mesh___Enderclem-HecateBiMod")
	// ```
	static void set_draw_remap(const std::string& source_entry, const std::string& target_entry)
	{
		// Source is a game-known name (Lookup works).
		// Target may be a custom variant name (needs StringIntern).
		static auto Lookup = *big::hades2_symbol_to_address["sgg::HashGuid::Lookup"]
		    .as_func<sgg::HashGuid*(sgg::HashGuid*, const char*, size_t)>();

		sgg::HashGuid src{};
		Lookup(&src, source_entry.c_str(), source_entry.size());
		sgg::HashGuid tgt = intern_string(target_entry);

		if (src.mId == 0 || tgt.mId == 0)
		{
			LOG(WARNING) << "draw: remap hash=0 for '"
			             << (src.mId == 0 ? source_entry : target_entry)
			             << "' — hash system not ready, skipping";
			return;
		}

		{
			std::unique_lock l(g_mutex);
			g_remap[src.mId] = tgt.mId;
		}
		update_active_flag();

		LOG(INFO) << "draw: remap " << source_entry << " -> " << target_entry
		          << " (hash " << src.mId << " -> " << tgt.mId << ")";
	}

	// Lua API: Function
	// Table: data
	// Name: set_draw_remap_hash
	// Param: source_hash: number: Source entry hash (from get_hash_guid_from_string).
	// Param: target_hash: number: Target entry hash.
	// Like set_draw_remap but takes pre-resolved numeric hashes.
	// Use when variant entry names aren't in the string intern table.
	static void set_draw_remap_hash(lua_Number source_hash, lua_Number target_hash)
	{
		auto src = (unsigned int)source_hash;
		auto tgt = (unsigned int)target_hash;

		{
			std::unique_lock l(g_mutex);
			g_remap[src] = tgt;
		}
		update_active_flag();

		LOG(INFO) << "draw: remap hash " << src << " -> " << tgt;
	}

	// Lua API: Function
	// Table: data
	// Name: clear_draw_remap
	// Param: source_entry: string: Original model entry name to clear remap for.
	// Reverts a model entry to draw its original mModelData entry.
	//
	// **Example Usage:**
	// ```lua
	// rom.data.clear_draw_remap("HecateBattle_Mesh")
	// ```
	static void clear_draw_remap(const std::string& source_entry)
	{
		static auto Lookup = *big::hades2_symbol_to_address["sgg::HashGuid::Lookup"]
		    .as_func<sgg::HashGuid*(sgg::HashGuid*, const char*, size_t)>();

		sgg::HashGuid src{};
		Lookup(&src, source_entry.c_str(), source_entry.size());

		if (src.mId == 0)
		{
			LOG(WARNING) << "draw: clear_remap hash=0 for '" << source_entry << "'";
			return;
		}

		{
			std::unique_lock l(g_mutex);
			g_remap.erase(src.mId);
		}
		update_active_flag();

		LOG(INFO) << "draw: clear remap " << source_entry << " (hash " << src.mId << ")";
	}

	// ─── v3.9 Option 4: GPU buffer memcpy ──────────────────────────────
	//
	// Data-flow:
	//   shader_byte = GrannyMeshData[+0x48]
	//   shader_eff  = gShaderEffects + shader_byte * 0xf28
	//   geo_idx     = *(uint32_t*)(shader_eff + 0x960)
	//   stride      = *(uint32_t*)(shader_eff + 0x95c)  (typically 40 for char vertex)
	//   geo_slot    = gStaticDrawBuffers.mpBegin + geo_idx * 72  (ForgeGeometryBuffers)
	//   vbuffer     = *(Buffer**)(geo_slot + 0x20)
	//   vbuffer_cpu = *(void**)(vbuffer + 0x00)           (persistent-mapped)
	//   ibuffer     = *(Buffer**)gStaticIndexBuffers
	//   ibuffer_cpu = *(void**)(ibuffer + 0x00)
	//
	//   Write at: vbuffer_cpu + vertex_handle * stride
	//             ibuffer_cpu + index_handle  * 2
	//
	// Static globals (resolved via PDB, survive game updates):
	//   sgg::gStaticDrawBuffers              — vector<ForgeGeometryBuffers>
	//   sgg::gShaderEffects                  — array, stride 0xf28
	//   sgg::gStaticIndexBuffers             — Buffer* (single global)

	// Plain-C helpers so we can use SEH (__try / __except) to survive bad pointers.
	// Each helper writes to *out_val and returns non-zero on success, 0 on fault.
	static int __stdcall safe_read_u64(const void* addr, uint64_t* out_val)
	{
		__try { *out_val = *(const uint64_t*)addr; return 1; }
		__except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
	}

	static int __stdcall safe_read_u32(const void* addr, uint32_t* out_val)
	{
		__try { *out_val = *(const uint32_t*)addr; return 1; }
		__except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
	}

	static int __stdcall safe_read_ptr(const void* addr, void** out_val)
	{
		__try { *out_val = *(void* const*)addr; return 1; }
		__except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
	}

	// Lua API: Function
	// Table: data
	// Name: dump_mesh_info
	// Param: entry_name: string: Entry name (e.g. "HecateBattle_Mesh").
	// Logs per-mesh buffer handles, shader, stride, and Buffer CPU pointers.
	// Used to validate the GPU-memcpy RE plan on a live character.
	// SEH-protected — reads that hit unmapped memory are logged as faults.
	static void dump_mesh_info(const std::string& entry_name)
	{
		LOG(INFO) << "dump_mesh_info: ENTER '" << entry_name << "'";

		static auto Lookup = *big::hades2_symbol_to_address["sgg::HashGuid::Lookup"]
		    .as_func<sgg::HashGuid*(sgg::HashGuid*, const char*, size_t)>();

		sgg::HashGuid guid{};
		Lookup(&guid, entry_name.c_str(), entry_name.size());
		LOG(INFO) << "  step1: hash=" << guid.mId;
		if (guid.mId == 0) { LOG(WARNING) << "  abort: hash=0 (hash table not ready or unknown name)"; return; }

		auto mdata_addr = big::hades2_symbol_to_address["sgg::Granny3D::mModelData"];
		auto sdb_addr   = big::hades2_symbol_to_address["sgg::gStaticDrawBuffers"];
		auto sef_addr   = big::hades2_symbol_to_address["sgg::gShaderEffects"];
		auto sib_addr   = big::hades2_symbol_to_address["sgg::gStaticIndexBuffers"];
		LOG(INFO) << "  step2: symbols"
		          << " mdata=" << HEX_TO_UPPER(mdata_addr.as<uintptr_t>())
		          << " sdb="   << HEX_TO_UPPER(sdb_addr.as<uintptr_t>())
		          << " sef="   << HEX_TO_UPPER(sef_addr.as<uintptr_t>())
		          << " sib="   << HEX_TO_UPPER(sib_addr.as<uintptr_t>());
		if (!mdata_addr || !sdb_addr || !sef_addr || !sib_addr)
		{ LOG(ERROR) << "  abort: one or more symbols missing"; return; }

		// Read hashtable header.
		uint8_t* mdata = mdata_addr.as<uint8_t*>();
		void* buckets_ptr = nullptr;
		uint64_t bucket_count = 0;
		if (!safe_read_ptr(mdata + 0x08, &buckets_ptr)) { LOG(ERROR) << "  fault reading mdata+0x08"; return; }
		if (!safe_read_u64(mdata + 0x10, &bucket_count)) { LOG(ERROR) << "  fault reading mdata+0x10"; return; }
		void** buckets = (void**)buckets_ptr;
		LOG(INFO) << "  step3: buckets=" << HEX_TO_UPPER((uintptr_t)buckets) << " count=" << bucket_count;
		if (!buckets || bucket_count == 0 || bucket_count > 0x100000)
		{ LOG(WARNING) << "  abort: mModelData empty or bogus"; return; }

		// EASTL hashtable bucket lookup (same formula as GetModelFile/LoadModelData).
		uint32_t h = guid.mId;
		h = ((h >> 16) ^ h) * 0x7feb352d;
		h = ((h >> 15) ^ h) * 0x846ca68b;
		h = (h >> 16) ^ h;
		uint64_t bucket_idx = h % bucket_count;
		void* head_ptr = nullptr;
		if (!safe_read_ptr(&buckets[bucket_idx], &head_ptr)) { LOG(ERROR) << "  fault reading bucket head"; return; }
		LOG(INFO) << "  step4: bucket_idx=" << bucket_idx << " head=" << HEX_TO_UPPER((uintptr_t)head_ptr);

		uint8_t* node = (uint8_t*)head_ptr;
		int walk_guard = 0;
		while (node && walk_guard++ < 32)
		{
			uint32_t node_id = 0;
			if (!safe_read_u32(node, &node_id)) { LOG(ERROR) << "  fault reading node->mId at " << HEX_TO_UPPER((uintptr_t)node); return; }
			if (node_id == guid.mId) break;
			void* next = nullptr;
			if (!safe_read_ptr(node + 0xC0, &next)) { LOG(ERROR) << "  fault reading node->mpNext"; return; }
			node = (uint8_t*)next;
		}
		if (!node || walk_guard >= 32) { LOG(WARNING) << "  abort: entry not found (walked " << walk_guard << ")"; return; }
		LOG(INFO) << "  step5: node=" << HEX_TO_UPPER((uintptr_t)node);

		// GrannyMeshData vector at node + 0x10 (mpBegin) / +0x18 (mpEnd).
		void* vec_begin_p = nullptr; void* vec_end_p = nullptr;
		if (!safe_read_ptr(node + 0x10, &vec_begin_p) || !safe_read_ptr(node + 0x18, &vec_end_p))
		{ LOG(ERROR) << "  fault reading GMD vector"; return; }
		uint8_t* vec_begin = (uint8_t*)vec_begin_p;
		uint8_t* vec_end   = (uint8_t*)vec_end_p;
		size_t mesh_count = (vec_end >= vec_begin) ? (size_t)(vec_end - vec_begin) / 0x50 : 0;
		LOG(INFO) << "  step6: vec=" << HEX_TO_UPPER((uintptr_t)vec_begin) << ".." << HEX_TO_UPPER((uintptr_t)vec_end)
		          << " meshes=" << mesh_count;
		if (mesh_count == 0 || mesh_count > 128) { LOG(WARNING) << "  abort: mesh_count out of range"; return; }

		uint8_t* sdb_vector = sdb_addr.as<uint8_t*>();
		uint8_t* sef_base   = sef_addr.as<uint8_t*>();
		uint8_t* sib_slot   = sib_addr.as<uint8_t*>();

		void* geo_first_p = nullptr;
		void* ibuffer_p   = nullptr;
		safe_read_ptr(sdb_vector, &geo_first_p);
		safe_read_ptr(sib_slot,   &ibuffer_p);
		uint8_t* geo_first = (uint8_t*)geo_first_p;
		uint8_t* ibuffer   = (uint8_t*)ibuffer_p;
		void*    ibuf_cpu  = nullptr;
		if (ibuffer) safe_read_ptr(ibuffer + 0x00, &ibuf_cpu);
		LOG(INFO) << "  step7: geo_first=" << HEX_TO_UPPER((uintptr_t)geo_first)
		          << " ibuffer=" << HEX_TO_UPPER((uintptr_t)ibuffer)
		          << " ibuf_cpu=" << HEX_TO_UPPER((uintptr_t)ibuf_cpu);

		for (size_t i = 0; i < mesh_count && i < 32; i++)
		{
			uint8_t* gmd = vec_begin + i * 0x50;
			uint32_t vh = 0, ih = 0, icount = 0, vcount = 0, mesh_type = 0;
			safe_read_u32(gmd + 0x30, &vh);
			safe_read_u32(gmd + 0x34, &ih);
			safe_read_u32(gmd + 0x38, &icount);
			safe_read_u32(gmd + 0x3C, &vcount);
			safe_read_u32(gmd + 0x4C, &mesh_type);

			// AddModelData's shader selection: type==2 → 0x22, else 0x13.
			uint8_t  shader_byte = (mesh_type == 2) ? 0x22 : 0x13;
			uint8_t* sef = sef_base + (uint64_t)shader_byte * 0xf28;
			uint32_t geo_idx = 0, stride = 0;
			safe_read_u32(sef + 0x960, &geo_idx);
			safe_read_u32(sef + 0x95c, &stride);

			uint8_t* geo_slot = geo_first ? (geo_first + (uint64_t)geo_idx * 72) : nullptr;
			void* vbuf_p = nullptr;
			if (geo_slot) safe_read_ptr(geo_slot + 0x20, &vbuf_p);
			uint8_t* vbuf = (uint8_t*)vbuf_p;
			void* vbuf_cpu = nullptr;
			if (vbuf) safe_read_ptr(vbuf + 0x00, &vbuf_cpu);

			// Read texture-binding fields at GMD+0x40 (name-hash gate) and +0x44 (handle).
			uint32_t tex_name_hash = 0, tex_handle = 0;
			safe_read_u32(gmd + 0x40, &tex_name_hash);
			safe_read_u32(gmd + 0x44, &tex_handle);

			LOG(INFO) << "  mesh[" << i << "] vh=" << vh << " ih=" << ih
			          << " vc=" << vcount << " ic=" << icount << " type=" << mesh_type
			          << " tex_name_hash=" << tex_name_hash
			          << " tex_handle=0x" << std::hex << tex_handle << std::dec
			          << " shader=0x" << std::hex << (int)shader_byte << std::dec
			          << " geo_idx=" << geo_idx << " stride=" << stride
			          << " vbuf=" << HEX_TO_UPPER((uintptr_t)vbuf)
			          << " vbuf_cpu=" << HEX_TO_UPPER((uintptr_t)vbuf_cpu);

			// One-shot diagnostic on mesh[0]: dump Buffer struct (first 0x50 bytes)
			// and a slice of its content at different candidate offsets.
			if (i == 0 && vbuf)
			{
				LOG(INFO) << "    --- Buffer* struct @" << HEX_TO_UPPER((uintptr_t)vbuf);
				for (int row = 0; row < 5; row++) {
					uint64_t q[4] = {};
					for (int k = 0; k < 4; k++) safe_read_u64(vbuf + row*32 + k*8, &q[k]);
					LOG(INFO) << "      +0x" << std::hex << (row*32) << std::dec
					          << ": " << HEX_TO_UPPER(q[0]) << " " << HEX_TO_UPPER(q[1])
					          << " " << HEX_TO_UPPER(q[2]) << " " << HEX_TO_UPPER(q[3]);
				}
				// Also scan the first 0x60 bytes of vbuf_cpu to see if there's structured data.
				if (vbuf_cpu) {
					LOG(INFO) << "    --- vbuf_cpu base @" << HEX_TO_UPPER((uintptr_t)vbuf_cpu);
					for (int row = 0; row < 4; row++) {
						uint32_t d[4] = {};
						for (int k = 0; k < 4; k++)
							safe_read_u32((uint8_t*)vbuf_cpu + row*16 + k*4, &d[k]);
						LOG(INFO) << "      +0x" << std::hex << (row*16) << std::dec
						          << ": " << HEX_TO_UPPER(d[0]) << " " << HEX_TO_UPPER(d[1])
						          << " " << HEX_TO_UPPER(d[2]) << " " << HEX_TO_UPPER(d[3]);
					}
				}
			}
		}
		LOG(INFO) << "dump_mesh_info: DONE";
	}

	// ─── Registration ─────────────────────────────────────────────────

	void bind(sol::state_view& state, sol::table& lua_ext)
	{
		auto ns = lua_ext["data"].get_or_create<sol::table>();
		ns.set_function("set_draw_visible", set_draw_visible);
		ns.set_function("set_draw_remap", set_draw_remap);
		ns.set_function("set_draw_remap_hash", set_draw_remap_hash);
		ns.set_function("clear_draw_remap", clear_draw_remap);
		ns.set_function("load_model_entry", load_model_entry);
		ns.set_function("dump_mesh_info", dump_mesh_info);

		// Lua API: Function
		// Table: data
		// Name: dump_pool_stats
		// Returns: integer — number of per-shader vertex buffers dumped.
		// Walks sgg::gStaticDrawBuffers and logs each shader-effect's
		// vertex pool: total capacity (Buffer.mSize &0xFFFFFFFF at +0x38),
		// cursor (+0x40 of ForgeGeometryBuffers — next-free vertex slot),
		// stride (from gShaderEffects[i].+0x95c).  Also logs the single
		// global gStaticIndexBuffers cursor + size.
		ns.set_function("dump_pool_stats", []() -> int {
			auto draw_buf_vec = big::hades2_symbol_to_address["sgg::gStaticDrawBuffers"];
			auto idx_buf_ptr  = big::hades2_symbol_to_address["sgg::gStaticIndexBuffers"];
			auto idx_off_ptr  = big::hades2_symbol_to_address["sgg::gStaticIndexBufferOffset"];
			auto shader_effects = big::hades2_symbol_to_address["sgg::gShaderEffects"];
			if (!draw_buf_vec || !idx_buf_ptr || !idx_off_ptr || !shader_effects)
			{
				LOG(ERROR) << "dump_pool_stats: symbols missing";
				return 0;
			}
			auto vec = draw_buf_vec.as<uint8_t *>();
			auto vec_begin = *(uint8_t **)(vec + 0x00);
			auto vec_end   = *(uint8_t **)(vec + 0x08);
			size_t count = (vec_end - vec_begin) / 72;
			LOG(INFO) << "=== Static vertex pool stats (" << count << " geo buffers) ===";
			// The shader→geo mapping isn't 1:1 (addShaderEffect calls
			// addStaticVertexBuffers twice with different sizes per effect),
			// so deriving the real per-shader byte stride from the geo
			// index alone is wrong.  Instead, report raw cursor + capacity,
			// plus a rough estimate at the common 40 B character-vertex
			// stride.  Only the raw cursor / capacity should be trusted.
			int logged = 0;
			for (size_t i = 0; i < count && i < 128; ++i)
			{
				uint8_t *geo = vec_begin + i * 72;
				uint8_t *buf = *(uint8_t **)(geo + 0x20);
				if (!buf) continue;
				uint32_t cursor = *(uint32_t *)(geo + 0x40);
				uint64_t msize_raw = *(uint64_t *)(buf + 0x38);
				uint32_t msize = (uint32_t)msize_raw;
				double cap_mb = msize / (1024.0 * 1024.0);
				double est_mb = (uint64_t)cursor * 40 / (1024.0 * 1024.0);
				LOG(INFO) << "  geo[" << i << "]: capacity=" << cap_mb
				          << " MB  cursor=" << cursor << " verts (~"
				          << est_mb << " MB @ 40B/vert)";
				logged++;
			}
			// Index buffer uses fixed 2-byte stride — bounds check is
			// exact, so this % is reliable (unlike the vertex-pool one).
			uint8_t *ibuf = *idx_buf_ptr.as<uint8_t **>();
			uint32_t ioff = *idx_off_ptr.as<uint32_t *>();
			if (ibuf)
			{
				uint64_t imsize_raw = *(uint64_t *)(ibuf + 0x38);
				uint32_t imsize = (uint32_t)imsize_raw;
				uint64_t iused = (uint64_t)ioff * 2;
				double ipct = imsize ? (100.0 * iused / imsize) : 0.0;
				LOG(INFO) << "  index buf: " << (iused / (1024.0 * 1024.0)) << " MB / "
				          << (imsize / (1024.0 * 1024.0)) << " MB (" << ipct << "%)";
			}
			return logged;
		});

		// Lua API: Function
		// Table: data
		// Name: populate_entry_textures
		// Param: entry_name: string
		// Returns: integer — number of textures populated.
		//
		// Walks the entry's GrannyMeshData vector and, for each type=0 mesh whose
		// GMD+0x40 (texture name hash) is non-zero, calls the game's own
		// sgg::GameAssetManager::GetTexture(hash, &out_handle) and writes the
		// result into GMD+0x44.  This mimics what ModelAnimation::PrepDraw does
		// for stock meshes — variants live alongside stock in mModelData but are
		// never walked by PrepDraw, so their GMD+0x44 stays 0 (→ white rendering
		// under hash remap).  Calling this once after loading a variant populates
		// its texture handles so DoDraw3D's fallback path can resolve them.
		ns.set_function("populate_entry_textures", [](const std::string& entry) -> int {
			static auto Lookup = *big::hades2_symbol_to_address["sgg::HashGuid::Lookup"]
			    .as_func<sgg::HashGuid*(sgg::HashGuid*, const char*, size_t)>();
			static auto GetTexture = big::hades2_symbol_to_address["sgg::GameAssetManager::GetTexture"]
			    .as_func<void(void*, uint32_t*, uint32_t)>();
			auto mdata_addr = big::hades2_symbol_to_address["sgg::Granny3D::mModelData"];
			if (!Lookup || !GetTexture || !mdata_addr)
			{
				LOG(ERROR) << "populate_entry_textures: required symbols missing";
				return 0;
			}

			sgg::HashGuid guid{};
			Lookup(&guid, entry.c_str(), entry.size());
			if (!guid.mId)
			{
				LOG(WARNING) << "populate_entry_textures: hash=0 for '" << entry << "'";
				return 0;
			}

			uint8_t* mdata = mdata_addr.as<uint8_t*>();
			void* buckets_ptr = nullptr;
			uint64_t bucket_count = 0;
			if (!safe_read_ptr(mdata + 0x08, &buckets_ptr)) return 0;
			if (!safe_read_u64(mdata + 0x10, &bucket_count)) return 0;
			if (!buckets_ptr || !bucket_count || bucket_count > 0x100000) return 0;

			uint32_t h = guid.mId;
			h = ((h >> 16) ^ h) * 0x7feb352d;
			h = ((h >> 15) ^ h) * 0x846ca68b;
			h = (h >> 16) ^ h;
			uint8_t* node = (uint8_t*)((void**)buckets_ptr)[h % bucket_count];
			int walk_guard = 0;
			while (node && walk_guard++ < 32)
			{
				uint32_t id = 0;
				if (!safe_read_u32(node, &id)) return 0;
				if (id == guid.mId) break;
				void* nxt = nullptr;
				if (!safe_read_ptr(node + 0xC0, &nxt)) return 0;
				node = (uint8_t*)nxt;
			}
			if (!node || walk_guard >= 32)
			{
				LOG(WARNING) << "populate_entry_textures: entry '" << entry << "' not found";
				return 0;
			}

			void* vb_p = nullptr; void* ve_p = nullptr;
			if (!safe_read_ptr(node + 0x10, &vb_p)) return 0;
			if (!safe_read_ptr(node + 0x18, &ve_p)) return 0;
			uint8_t* vec_begin = (uint8_t*)vb_p;
			uint8_t* vec_end   = (uint8_t*)ve_p;
			size_t mesh_count = (vec_end >= vec_begin) ? (size_t)(vec_end - vec_begin) / 0x50 : 0;
			if (mesh_count == 0 || mesh_count > 128) return 0;

			int populated = 0;
			for (size_t i = 0; i < mesh_count; i++)
			{
				uint8_t* gmd = vec_begin + i * 0x50;

				uint32_t mesh_type=0, name_hash=0, old_handle=0;
				safe_read_u32(gmd + 0x4C, &mesh_type);
				safe_read_u32(gmd + 0x40, &name_hash);
				safe_read_u32(gmd + 0x44, &old_handle);

				// Game's PrepDraw path only fills GMD+0x44 for type=0 meshes.
				// Mirror that guard; outlines (type=1) and shadows (type=2)
				// use different resolution paths and we don't want to touch them.
				if (mesh_type != 0) continue;
				if (name_hash == 0) continue;

				uint32_t new_handle = 0;
				GetTexture(nullptr, &new_handle, name_hash);
				*(uint32_t*)(gmd + 0x44) = new_handle;

				LOG(INFO) << "populate_entry_textures: '" << entry << "' mesh[" << i << "]"
				          << " name_hash=" << name_hash
				          << " handle 0x" << std::hex << old_handle << " -> 0x" << new_handle << std::dec;
				populated++;
			}
			LOG(INFO) << "populate_entry_textures: '" << entry << "' populated " << populated << " handle(s)";
			return populated;
		});

		// Lua API: Function
		// Table: data
		// Name: set_mesh_visible
		// Param: entry_name: string: Model entry (e.g. "HecateHub_Mesh").
		// Param: mesh_name: string: Mesh name inside that entry (e.g. "TorusHubMesh").
		// Param: visible: boolean: true to show, false to hide.
		// Returns: boolean — true on success.
		//
		// Finer-grained than set_draw_visible (which hides the whole entry):
		// walks the entry's GrannyMeshData vector, finds the mesh whose
		// mesh-name hash at GMD+0x48 matches `mesh_name`, and flips its
		// texture-name hash at GMD+0x40 between 0 (hide) and the original
		// value (show).
		//
		// Hide path uses DoDraw3D's OWN mesh-type switch: setting
		// GMD+0x4C = 2 makes the main-draw function skip to
		// next-iteration at 0x1401ebd25 — no cmdDrawIndexed is issued,
		// no texture lookup attempted.  This is the same branch the
		// engine takes for its own shadow meshes (they're drawn via
		// DoDrawShadow3D instead, which our accessory meshes don't
		// have a shadow entry in, so they stay hidden everywhere).
		// No DX12 validation errors, no command-list poisoning.
		//
		// Used for instant accessory toggle: mesh_add mods merge their
		// meshes INTO stock entries, so the entry-level draw-gate would
		// hide the body alongside the accessory.  Per-mesh visibility
		// keeps the body on and only suppresses the accessory meshes.
		ns.set_function("set_mesh_visible", [](const std::string& entry,
		                                        const std::string& mesh_name,
		                                        bool visible) -> bool {
			// Saved mesh_type keyed by (entry_hash, mesh_hash, idx).  Index
			// distinguishes multiple GMDs sharing the same name hash
			// (e.g. main+outline+shadow variants under a shared name, or
			// duplicate GLB meshes split by material).  Using raw GMD
			// pointer would be invalidated if the GMD vector ever
			// reallocates — unlikely in Hades II's static-load model but
			// not guaranteed by the engine contract.  std::map so we
			// don't have to write a tuple hasher.
			using SavedKey = std::tuple<uint32_t, uint32_t, size_t>;
			static std::map<SavedKey, uint8_t> g_saved_mesh_type;
			static std::mutex g_saved_mutex;

			static auto Lookup = *big::hades2_symbol_to_address["sgg::HashGuid::Lookup"]
			    .as_func<sgg::HashGuid*(sgg::HashGuid*, const char*, size_t)>();
			auto mdata_addr = big::hades2_symbol_to_address["sgg::Granny3D::mModelData"];
			if (!Lookup || !mdata_addr)
			{
				LOG(ERROR) << "set_mesh_visible: required symbols missing";
				return false;
			}

			sgg::HashGuid entry_guid{};
			Lookup(&entry_guid, entry.c_str(), entry.size());
			if (!entry_guid.mId)
			{
				LOG(WARNING) << "set_mesh_visible: entry hash=0 for '" << entry << "'";
				return false;
			}
			sgg::HashGuid mesh_guid{};
			Lookup(&mesh_guid, mesh_name.c_str(), mesh_name.size());
			if (!mesh_guid.mId)
			{
				LOG(WARNING) << "set_mesh_visible: mesh hash=0 for '" << mesh_name << "'";
				return false;
			}

			uint8_t* mdata = mdata_addr.as<uint8_t*>();
			void* buckets_ptr = nullptr;
			uint64_t bucket_count = 0;
			if (!safe_read_ptr(mdata + 0x08, &buckets_ptr)) return false;
			if (!safe_read_u64(mdata + 0x10, &bucket_count)) return false;
			if (!buckets_ptr || !bucket_count || bucket_count > 0x100000) return false;

			uint32_t h = entry_guid.mId;
			h = ((h >> 16) ^ h) * 0x7feb352d;
			h = ((h >> 15) ^ h) * 0x846ca68b;
			h = (h >> 16) ^ h;
			uint8_t* node = (uint8_t*)((void**)buckets_ptr)[h % bucket_count];
			int walk_guard = 0;
			while (node && walk_guard++ < 32)
			{
				uint32_t id = 0;
				if (!safe_read_u32(node, &id)) return false;
				if (id == entry_guid.mId) break;
				void* nxt = nullptr;
				if (!safe_read_ptr(node + 0xC0, &nxt)) return false;
				node = (uint8_t*)nxt;
			}
			if (!node || walk_guard >= 32)
			{
				LOG(WARNING) << "set_mesh_visible: entry '" << entry << "' not in mModelData";
				return false;
			}

			void* vb_p = nullptr; void* ve_p = nullptr;
			if (!safe_read_ptr(node + 0x10, &vb_p)) return false;
			if (!safe_read_ptr(node + 0x18, &ve_p)) return false;
			uint8_t* vec_begin = (uint8_t*)vb_p;
			uint8_t* vec_end   = (uint8_t*)ve_p;
			size_t mesh_count = (vec_end >= vec_begin) ? (size_t)(vec_end - vec_begin) / 0x50 : 0;
			if (mesh_count == 0 || mesh_count > 128) return false;

			// Sentinel byte used for hidden state (= shadow mesh type,
			// which DoDraw3D skips to next-iteration).
			constexpr uint8_t HIDE_TYPE = 2;
			int matched = 0;
			std::lock_guard lk(g_saved_mutex);
			for (size_t i = 0; i < mesh_count; i++)
			{
				uint8_t* gmd = vec_begin + i * 0x50;
				uint32_t gmd_mesh_hash = 0;
				if (!safe_read_u32(gmd + 0x48, &gmd_mesh_hash)) continue;
				if (gmd_mesh_hash != mesh_guid.mId) continue;

				uint8_t current_type = *(uint8_t*)(gmd + 0x4C);
				SavedKey key{entry_guid.mId, mesh_guid.mId, i};

				if (visible)
				{
					auto it = g_saved_mesh_type.find(key);
					if (it != g_saved_mesh_type.end())
					{
						*(uint8_t*)(gmd + 0x4C) = it->second;
						g_saved_mesh_type.erase(it);
					}
					// else: already visible — no-op
				}
				else
				{
					if (current_type != HIDE_TYPE && !g_saved_mesh_type.count(key))
					{
						g_saved_mesh_type[key] = current_type;
						*(uint8_t*)(gmd + 0x4C) = HIDE_TYPE;
					}
					// else: already hidden — no-op
				}
				matched++;
				// Don't break: continue so main+outline+shadow variants
				// with the same mesh-name hash all get toggled together.
			}
			if (matched == 0)
			{
				LOG(WARNING) << "set_mesh_visible: mesh '" << mesh_name
				             << "' not in entry '" << entry << "'";
				return false;
			}
			LOG(INFO) << "set_mesh_visible: " << entry << "/" << mesh_name
			          << " -> " << (visible ? "show" : "hide")
			          << " (" << matched << " mesh" << (matched > 1 ? "es" : "") << ")";
			return true;
		});

		// Lua API: Function
		// Table: data
		// Name: swap_to_variant
		// Param: stock_entry: string: Stock entry name (e.g. "HecateHub_Mesh").
		// Param: variant_entry: string: Variant entry name loaded in mModelData.
		// Returns: boolean — true on success.
		//
		// One-call atomic outfit switch, matching the engine's own rendering
		// architecture:
		//   1. Populate the variant's GMD+0x44 via sgg::GameAssetManager::GetTexture
		//      (mirrors what ModelAnimation::PrepDraw does for stock entries).
		//   2. Install a hash remap so draw commands using `stock_entry` get
		//      redirected to the variant entry at dispatch time.
		//
		// Variant textures resolve through the variant's own `GMD+0x40`
		// (texture name hashes written by AddModelData).  No vcount/topology
		// constraints; stock state is never mutated.
		//
		// **Example Usage:**
		// ```lua
		// rom.data.swap_to_variant("HecateHub_Mesh", "Enderclem_HecateBiMod_V0_Mesh")
		// -- later:
		// rom.data.restore_stock("HecateHub_Mesh")
		// ```
		ns.set_function("swap_to_variant", [](const std::string& stock_entry,
		                                       const std::string& variant_entry) -> bool {
			// Just install the hash remap.  Texture handles (GMD+0x44) must be
			// populated ahead of time via populate_entry_textures at a known-safe
			// window (first ImGui frame, after LoadAllModelAndAnimationData).
			// Calling GetTexture from the render-thread ImGui callback while the
			// game thread is actively rendering the target entry deadlocks the
			// render pipeline (observed: "Waiting for RenderCommands to be ready
			// to write").  Keeping this path to a single atomic map write makes
			// swaps cheap and thread-safe.
			static auto Lookup = *big::hades2_symbol_to_address["sgg::HashGuid::Lookup"]
			    .as_func<sgg::HashGuid*(sgg::HashGuid*, const char*, size_t)>();
			if (!Lookup) { LOG(ERROR) << "swap_to_variant: Lookup missing"; return false; }

			sgg::HashGuid variant_guid{};
			Lookup(&variant_guid, variant_entry.c_str(), variant_entry.size());
			if (!variant_guid.mId)
			{
				LOG(WARNING) << "swap_to_variant: variant hash=0 ('" << variant_entry << "')";
				return false;
			}
			sgg::HashGuid stock_guid{};
			Lookup(&stock_guid, stock_entry.c_str(), stock_entry.size());
			if (!stock_guid.mId)
			{
				LOG(WARNING) << "swap_to_variant: stock hash=0 ('" << stock_entry << "')";
				return false;
			}

			{
				std::unique_lock l(g_mutex);
				g_remap[stock_guid.mId] = variant_guid.mId;
			}
			update_active_flag();

			LOG(INFO) << "swap_to_variant: '" << stock_entry << "' -> '" << variant_entry << "'";
			return true;
		});

		// Lua API: Function
		// Table: data
		// Name: restore_stock
		// Param: stock_entry: string: Stock entry name to revert to.
		// Returns: boolean — true on success.
		// Clears any active hash remap for the given stock entry.  No-op if no
		// remap is active.  Populated variant GMD+0x44 handles are left in place
		// (they're harmless — only used when a remap is active).
		ns.set_function("restore_stock", [](const std::string& stock_entry) -> bool {
			static auto Lookup = *big::hades2_symbol_to_address["sgg::HashGuid::Lookup"]
			    .as_func<sgg::HashGuid*(sgg::HashGuid*, const char*, size_t)>();
			sgg::HashGuid stock{};
			Lookup(&stock, stock_entry.c_str(), stock_entry.size());
			if (!stock.mId)
			{
				LOG(WARNING) << "restore_stock: hash=0 ('" << stock_entry << "')";
				return false;
			}
			{
				std::unique_lock l(g_mutex);
				g_remap.erase(stock.mId);
			}
			update_active_flag();
			LOG(INFO) << "restore_stock: '" << stock_entry << "' remap cleared";
			return true;
		});

		// NOTE: No hook on LoadAllModelAndAnimationData — a second detour
		// are loaded automatically by the game when add_granny_file exposes
		// them.  The double-hook was causing weapon model corruption.

		// Detour hooks on DoDraw3D, DoDrawShadow3D, DoDraw3DThumbnail.
		{
			auto addr = big::hades2_symbol_to_address["sgg::DrawManager::DoDraw3D"];
			if (addr)
			{
				static auto hook_ = big::hooking::detour_hook_helper::add<hook_DoDraw3D>(
				    "drawDoDraw3D", addr);
				LOG(INFO) << "draw: hooked DoDraw3D";
			}
		}
		{
			auto addr = big::hades2_symbol_to_address["sgg::DrawManager::DoDrawShadow3D"];
			if (addr)
			{
				static auto hook_ = big::hooking::detour_hook_helper::add<hook_DoDrawShadow3D>(
				    "drawDoDrawShadow3D", addr);
				LOG(INFO) << "draw: hooked DoDrawShadow3D";
			}
		}
		{
			auto addr = big::hades2_symbol_to_address["sgg::DrawManager::DoDraw3DThumbnail"];
			if (addr)
			{
				static auto hook_ = big::hooking::detour_hook_helper::add<hook_DoDraw3DThumbnail>(
				    "drawDoDraw3DThumbnail", addr);
				LOG(INFO) << "draw: hooked DoDraw3DThumbnail";
			}
		}

		// Manual code cave for DoDrawShadowCast3D.
		{
			auto addr = big::hades2_symbol_to_address["sgg::DrawManager::DoDraw3D"];
			if (addr)
				install_shadow_cast_patch(addr.as<uintptr_t>());
		}
	}

} // namespace lua::hades::draw
