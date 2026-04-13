/// @file draw.cpp
/// @brief Draw-call control for 3D model entries.
///
/// Hooks sgg::DrawManager's four draw functions to suppress draw calls
/// for entries whose HashGuid is in a hidden set.  Provides the Lua
/// binding `rom.data.set_draw_visible(entry_name, visible)`.
///
/// Three functions (DoDraw3D, DoDrawShadow3D, DoDraw3DThumbnail) share
/// the PDB signature `static void(const vector<RenderMesh*>&, uint, int, HashGuid)`
/// and are handled with standard detour hooks on param4.
///
/// DoDrawShadowCast3D has a different signature (no HashGuid param).
/// The entry hash IS present in the draw entry at [r10+0x28] but the
/// dispatch code skips reading it for the shadow path.  SafetyHook
/// mid-hooks fail in this dispatch area, so we patch the bytes directly
/// with VirtualProtect + a hand-assembled code cave.

#include "draw.hpp"

#include "hades_ida.hpp"

#include <hades2/pdb_symbol_map.hpp>
#include <hooks/hooking.hpp>
#include <lua/lua_manager.hpp>
#include <memory/gm_address.hpp>
#include <shared_mutex>
#include <string/string.hpp>

namespace lua::hades::draw
{
	// ─── State ─────────────────────────────────────────────────────────

	static std::shared_mutex g_mutex;
	static std::unordered_set<unsigned int> g_hidden_entries;

	// Fast-path flag for the code cave — avoids the function-call overhead
	// on every draw entry when nothing is hidden.
	static volatile uint8_t g_any_active = 0;

	// Called from the code cave via function pointer.
	static bool is_hash_hidden(uint32_t hash)
	{
		std::shared_lock l(g_mutex);
		return g_hidden_entries.count(hash) > 0;
	}

	// ─── Detour hooks (DoDraw3D, DoDrawShadow3D, DoDraw3DThumbnail) ──

	static void hook_DoDraw3D(void* vec_ref, unsigned int index, int param, sgg::HashGuid hash)
	{
		{
			std::shared_lock l(g_mutex);
			if (g_hidden_entries.count(hash.mId))
				return;
		}
		big::g_hooking->get_original<hook_DoDraw3D>()(vec_ref, index, param, hash);
	}

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
		*(uintptr_t*)(data_base + 0x08) = (uintptr_t)&is_hash_hidden;
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

		// Fast path: if nothing is hidden, skip straight to original code
		emit({0x48, 0x8B, 0x05}); emit_rel32(rip_data(0x00, 4)); // mov rax, [rip+g_any_active]
		emit({0x80, 0x38, 0x00});                                  // cmp byte [rax], 0
		emit({0x74}); size_t je_fast = cur(); emit({0x00});         // je .not_hidden

		// Slow path: save regs, call is_hash_hidden(ecx = [r10+0x28])
		emit({0x51, 0x52});                                         // push rcx; push rdx
		emit({0x48, 0x83, 0xEC, 0x28});                             // sub rsp, 0x28
		emit({0x41, 0x8B, 0x4A, 0x28});                             // mov ecx, [r10+0x28]
		emit({0x48, 0x8B, 0x05}); emit_rel32(rip_data(0x08, 4));   // mov rax, [rip+is_hash_hidden]
		emit({0xFF, 0xD0});                                         // call rax
		emit({0x48, 0x83, 0xC4, 0x28});                             // add rsp, 0x28
		emit({0x5A, 0x59});                                         // pop rdx; pop rcx
		emit({0x84, 0xC0});                                         // test al, al
		emit({0x75}); size_t jnz_skip = cur(); emit({0x00});        // jnz .skip

		// .not_hidden — replay original instructions
		size_t not_hidden = cur();
		*((uint8_t*)cave + je_fast) = (uint8_t)(not_hidden - je_fast - 1);
		emit({0x41, 0x80, 0x7A, 0x2D, 0x00});                      // cmp byte [r10+0x2d], 0
		emit({0x74}); size_t je_main = cur(); emit({0x00});          // je .main
		emit({0xFF, 0x25}); emit_rel32(rip_data(0x10, 4));          // jmp [shadow_continue]

		// .main — non-shadow path
		size_t main_off = cur();
		*((uint8_t*)cave + je_main) = (uint8_t)(main_off - je_main - 1);
		emit({0xFF, 0x25}); emit_rel32(rip_data(0x18, 4));          // jmp [main_continue]

		// .skip — entry is hidden, advance loop
		size_t skip_off = cur();
		*((uint8_t*)cave + jnz_skip) = (uint8_t)(skip_off - jnz_skip - 1);
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

		g_any_active = g_hidden_entries.empty() ? 0 : 1;

		LOG(INFO) << "draw: " << entry_name
		          << (visible ? " SHOW" : " HIDE")
		          << " (hash=" << guid.mId
		          << ", set " << old_size << " -> " << new_size << ")";
	}

	// ─── Registration ─────────────────────────────────────────────────

	void bind(sol::state_view& state, sol::table& lua_ext)
	{
		auto ns = lua_ext["data"].get_or_create<sol::table>();
		ns.set_function("set_draw_visible", set_draw_visible);

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
