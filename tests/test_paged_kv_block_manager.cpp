#include "llm/paged_kv_block_manager.h"

#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) {
  throw std::runtime_error("test_paged_kv_block_manager: " + message);
}

static void expect_throw(const std::function<void()>& fn, const std::string& name) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  fail(name + " did not throw");
}

static void check_invariants(const BlockManager& manager) {
  if (manager.free_block_count() + manager.used_block_count() !=
      manager.total_block_count())
    fail("free + used != total");
}

int main() {
  try {
    expect_throw([] { BlockManager invalid(0); }, "zero-sized manager");
    BlockManager manager(8);
    if (manager.total_block_count() != 8 || manager.free_block_count() != 8 ||
        manager.used_block_count() != 0)
      fail("initial counts mismatch");
    check_invariants(manager);

    const BlockId a = manager.allocate();
    if (!manager.is_allocated(a) || manager.ref_count(a) != 1 ||
        manager.free_block_count() != 7)
      fail("allocate/refcount mismatch");
    manager.retain(a);
    if (manager.ref_count(a) != 2) fail("retain mismatch");
    manager.release(a);
    if (manager.ref_count(a) != 1) fail("first release mismatch");
    manager.release(a);
    if (manager.is_allocated(a) || manager.ref_count(a) != 0 ||
        manager.free_block_count() != 8)
      fail("last release did not return block");
    expect_throw([&] { manager.release(a); }, "double release");
    expect_throw([&] { manager.retain(a); }, "retain free block");
    expect_throw([&] { manager.ref_count(kInvalidBlockId); }, "invalid id");
    std::vector<BlockId> exhausted;
    for (int i = 0; i < 8; ++i) exhausted.push_back(manager.allocate());
    expect_throw([&] { manager.allocate(); }, "pool exhaustion");
    for (BlockId id : exhausted) manager.release(id);
    check_invariants(manager);
    std::cout << "BlockManager allocate/retain/release and rejection paths passed\n";

    BlockManager mapping_manager(32);
    const std::vector<std::size_t> appends{1, 15, 16, 17, 32, 33};
    const std::vector<std::size_t> expected_blocks{1, 1, 1, 2, 2, 3};
    for (std::size_t i = 0; i < appends.size(); ++i) {
      SequenceBlockTable sequence(mapping_manager, 16, appends[i]);
      sequence.append_tokens(appends[i]);
      if (sequence.block_count() != expected_blocks[i] || sequence.token_count() != appends[i])
        fail("block boundary count mismatch");
      for (std::size_t token = 0; token < sequence.token_count(); ++token) {
        const PhysicalLocation location = sequence.locate(token);
        if (location.block_id != sequence.block_table().at(token / 16) ||
            location.offset != token % 16)
          fail("logical-to-physical mapping mismatch");
      }
      expect_throw([&] { sequence.locate(appends[i]); }, "locate out of range");
    }
    SequenceBlockTable sequence(mapping_manager, 16, 65);
    sequence.append_tokens(33);
    const auto table_before = sequence.block_table();
    const auto tokens_before = sequence.token_count();
    expect_throw([&] { sequence.append_tokens(33); }, "max_tokens append");
    if (sequence.token_count() != tokens_before || sequence.block_table() != table_before)
      fail("max_tokens failure changed sequence state");
    std::cout << "block_size=16 append 1/15/16/17/32/33 and mapping passed\n";

    BlockManager isolated_manager(8);
    SequenceBlockTable first_sequence(isolated_manager, 4, 8);
    SequenceBlockTable second_sequence(isolated_manager, 4, 8);
    first_sequence.append_tokens(5);
    second_sequence.append_tokens(5);
    const auto first_table = first_sequence.block_table();
    const auto second_table = second_sequence.block_table();
    std::set<BlockId> first_ids(first_table.begin(), first_table.end());
    std::set<BlockId> second_ids(second_table.begin(), second_table.end());
    if (first_ids.size() != first_table.size() || second_ids.size() != second_table.size())
      fail("a sequence block table contains duplicate physical blocks");
    for (BlockId id : first_table) {
      if (second_ids.count(id) != 0 || !isolated_manager.is_allocated(id) ||
          isolated_manager.ref_count(id) != 1)
        fail("live sequence block tables overlap or have unexpected refcount");
    }
    for (BlockId id : second_table) {
      if (!isolated_manager.is_allocated(id) || isolated_manager.ref_count(id) != 1)
        fail("second live sequence block has unexpected allocation state");
    }
    first_sequence.release_all();
    for (BlockId id : first_table)
      if (isolated_manager.is_allocated(id) || isolated_manager.ref_count(id) != 0)
        fail("first sequence block was not returned after release_all");
    for (BlockId id : second_table)
      if (!isolated_manager.is_allocated(id) || isolated_manager.ref_count(id) != 1)
        fail("releasing one sequence affected another");
    second_sequence.release_all();
    second_sequence.release_all();
    if (isolated_manager.used_block_count() != 0) fail("release_all leaked blocks");
    std::cout << "sequence isolation and idempotent release_all passed\n";

    BlockManager incremental_manager(8);
    SequenceBlockTable incremental(incremental_manager, 16, 64);
    const auto check_incremental_counts = [&](std::size_t expected_blocks) {
      if (incremental.token_count() == 0 || incremental.block_count() != expected_blocks ||
          incremental_manager.used_block_count() != expected_blocks ||
          incremental_manager.free_block_count() !=
              incremental_manager.total_block_count() - expected_blocks)
        fail("incremental append block manager counts mismatch");
      for (BlockId id : incremental.block_table())
        if (!incremental_manager.is_allocated(id) || incremental_manager.ref_count(id) != 1)
          fail("incremental append block has unexpected allocation state");
    };
    incremental.append_tokens(15);
    check_incremental_counts(1);
    const BlockId first_block = incremental.block_table().at(0);
    incremental.append_tokens(1);
    check_incremental_counts(1);
    if (incremental.block_table().at(0) != first_block ||
        incremental.locate(15).block_id != first_block || incremental.locate(15).offset != 15)
      fail("append into the tail block did not reuse block/offset 15");
    incremental.append_tokens(1);
    check_incremental_counts(2);
    const BlockId second_block = incremental.block_table().at(1);
    if (incremental.token_count() != 17 || incremental.locate(16).block_id != second_block ||
        incremental.locate(16).offset != 0 || second_block == first_block)
      fail("crossing the first block boundary mismatch");
    incremental.append_tokens(15);
    check_incremental_counts(2);
    if (incremental.token_count() != 32 || incremental.block_table().at(0) != first_block ||
        incremental.block_table().at(1) != second_block)
      fail("append into the second tail block changed existing IDs");
    incremental.append_tokens(1);
    check_incremental_counts(3);
    if (incremental.token_count() != 33 || incremental.block_table().at(2) == first_block ||
        incremental.block_table().at(2) == second_block || incremental.locate(32).offset != 0)
      fail("third block allocation mismatch");
    incremental.release_all();
    if (incremental_manager.used_block_count() != 0 ||
        incremental_manager.free_block_count() != incremental_manager.total_block_count())
      fail("incremental release_all did not return all blocks");
    std::cout << "incremental append tail reuse and block ID stability passed\n";

    BlockManager rollback_manager(2);
    SequenceBlockTable rollback(rollback_manager, 4, 12);
    rollback.append_tokens(4);
    const auto rollback_table = rollback.block_table();
    const std::size_t rollback_free = rollback_manager.free_block_count();
    expect_throw([&] { rollback.append_tokens(9); }, "transactional pool exhaustion");
    if (rollback.token_count() != 4 || rollback.block_table() != rollback_table ||
        rollback_manager.free_block_count() != rollback_free ||
        rollback_manager.used_block_count() != 1 ||
        rollback_manager.ref_count(rollback_table.front()) != 1)
      fail("multi-block append did not roll back completely");
    rollback.release_all();
    if (rollback_manager.free_block_count() != 2) fail("rollback manager did not recover");
    std::cout << "multi-block append transactional rollback passed\n";

    {
      BlockManager destructor_manager(16);
      {
        SequenceBlockTable temporary(destructor_manager, 4, 16);
        temporary.append_tokens(9);
        if (destructor_manager.used_block_count() != 3) fail("destructor setup mismatch");
      }
      if (destructor_manager.used_block_count() != 0 ||
          destructor_manager.free_block_count() != destructor_manager.total_block_count())
        fail("destructor did not release all blocks");
    }
    std::cout << "destructor automatic reclamation passed\n";

    BlockManager stress_manager(64);
    // SequenceBlockTable is intentionally non-movable, so exercise a stable
    // set of independent tables through individually scoped deterministic loops.
    for (int cycle = 0; cycle < 40; ++cycle) {
      {
        SequenceBlockTable s1(stress_manager, 8, 32);
        SequenceBlockTable s2(stress_manager, 8, 32);
        s1.append_tokens(static_cast<std::size_t>((cycle * 3) % 25));
        s2.append_tokens(static_cast<std::size_t>((cycle * 5 + 1) % 25));
        check_invariants(stress_manager);
        if (s1.block_count() > 0 && s2.block_count() > 0 &&
            std::set<BlockId>(s1.block_table().begin(), s1.block_table().end()).size() !=
                s1.block_count())
          fail("duplicate block in stress sequence");
      }
      check_invariants(stress_manager);
      if (stress_manager.used_block_count() != 0) fail("stress cycle leaked blocks");
    }
    std::cout << "deterministic multi-sequence stress invariants passed\n";
    std::cout << "test_paged_kv_block_manager passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
