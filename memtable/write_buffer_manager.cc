//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "rocksdb/write_buffer_manager.h"

#include <memory>
#include "cache/cache_entry_roles.h"
#include "cache/cache_reservation_manager.h"
#include "db/db_impl/db_impl.h"
#include "rocksdb/status.h"
#include "util/coding.h"
#include "rocksdb/tg_thread_local.h"

#include <execinfo.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include <vector>
#include <mutex>

namespace ROCKSDB_NAMESPACE {

WriteBufferManager::WriteBufferManager(size_t buffer_size,
                                       std::shared_ptr<Cache> cache,
                                       bool allow_stall,
                                       size_t num_clients)
    : buffer_size_(buffer_size),
      mutable_limit_(buffer_size),
      memory_used_(0),
      per_client_memory_used_(num_clients),
      per_client_buffer_size_(num_clients, buffer_size),
      memory_active_(0),
      cache_res_mgr_(nullptr),
      per_client_queue_(num_clients),
      allow_stall_(allow_stall),
      per_client_stall_active_(num_clients),
      per_client_stall_count_(num_clients),  // Initialize stall count for clients
      total_stall_count_(0) {
        
  std::cout << "[FAIRDB_LOG] WBM Size: " << buffer_size << std::endl;
  for (size_t i = 0; i < num_clients; ++i) {
    per_client_memory_used_[i] = 0;
    per_client_stall_active_[i] = false;
    per_client_stall_count_[i] = 0;
  }

  if (cache) {
    // Memtable's memory usage tends to fluctuate frequently
    // therefore we set delayed_decrease = true to save some dummy entry
    // insertion on memory increase right after memory decrease
    cache_res_mgr_ = std::make_shared<
        CacheReservationManagerImpl<CacheEntryRole::kWriteBuffer>>(
        cache, true /* delayed_decrease */);
  }

  mt_log_file_.open("logs/memtable_stats.txt", std::ios::out | std::ios::trunc);
  // Ensure log file is open and ready for use
  if (!mt_log_file_.is_open()) {
    throw std::ios_base::failure("Failed to open log/memtable_stats.txt");
  }
}

WriteBufferManager::~WriteBufferManager() {
#ifndef NDEBUG
  std::unique_lock<std::mutex> lock(mu_);
  for (const auto& queue : per_client_queue_) {
    assert(queue.empty());
  }
#endif
  if (mt_log_file_.is_open()) {
    mt_log_file_.close();
  }
}

std::size_t WriteBufferManager::dummy_entries_in_cache_usage() const {
  if (cache_res_mgr_ != nullptr) {
    return cache_res_mgr_->GetTotalReservedCacheSize();
  } else {
    return 0;
  }
}

void PrintStackTrace() {
    constexpr int max_frames = 64;
    void* addrlist[max_frames];

    // Retrieve the current stack addresses
    int addrlen = backtrace(addrlist, max_frames);

    if (addrlen == 0) {
        std::cerr << "No stack trace available\n";
        return;
    }

    // Convert the addresses into readable symbols
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    // Print out all the frames
    std::cout << "Stack trace:\n";
    for (int i = 0; i < addrlen; ++i) {
        std::cout << symbollist[i] << '\n';
    }

    free(symbollist);  // `backtrace_symbols` uses malloc internally
}

bool WriteBufferManager::ShouldStall(int client_id) const {
  if (client_id < 0) {
    std::cout << "[FAIRDB_LOG] Unaccounted ShouldStall " << client_id << std::endl;
    return false;
  }
  if (!allow_stall_.load(std::memory_order_relaxed) || !enabled()) {
    return false;
  }

  return IsStallActive(client_id) || IsStallThresholdExceeded(client_id);
}

bool WriteBufferManager::IsStallThresholdExceeded(int client_id) const {
    return per_client_memory_used_[client_id].load(std::memory_order_relaxed) >= per_client_buffer_size_[client_id];

  // return (per_client_memory_usage(client_id) >= per_client_buffer_size_[client_id]) || (memory_usage() >= buffer_size_);
}

void WriteBufferManager::SetPerClientBufferSize(int client_id, size_t buffer_size) {
  per_client_buffer_size_[client_id] = buffer_size;
}

void WriteBufferManager::ReserveMem(size_t mem) {
  int client_id = TG_GetThreadMetadata().client_id;
  if (client_id < 0) {
    std::cout << "[FAIRDB_LOG] Unaccounted ReserveMem " << client_id << std::endl;
    return;
  }
  if (cache_res_mgr_ != nullptr) {
    ReserveMemWithCache(mem);
  } else if (enabled()) {
    memory_used_.fetch_add(mem, std::memory_order_relaxed);
    per_client_memory_used_[client_id].fetch_add(mem, std::memory_order_relaxed);
  }
  if (enabled()) {
    memory_active_.fetch_add(mem, std::memory_order_relaxed);
  }
  mt_log_file_ << "wbm," << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << "," << client_id << ",res," << mem << "," << per_client_memory_used_[client_id].load(std::memory_order_relaxed) << std::endl;
}

// Should only be called from write thread
void WriteBufferManager::ReserveMemWithCache(size_t mem) {
  assert(cache_res_mgr_ != nullptr);
  // Use a mutex to protect various data structures. Can be optimized to a
  // lock-free solution if it ends up with a performance bottleneck.
  std::lock_guard<std::mutex> lock(cache_res_mgr_mu_);

  size_t new_mem_used = memory_usage() + mem;
  memory_used_.store(new_mem_used, std::memory_order_relaxed);
  Status s = cache_res_mgr_->UpdateCacheReservation(new_mem_used);

  // We absorb the error since WriteBufferManager is not able to handle
  // this failure properly. Ideallly we should prevent this allocation
  // from happening if this cache charging fails.
  // [TODO] We'll need to improve it in the future and figure out what to do on
  // error
  s.PermitUncheckedError();
}

void WriteBufferManager::ScheduleFreeMem(size_t mem) {
  if (enabled()) {
    memory_active_.fetch_sub(mem, std::memory_order_relaxed);
  }
}

void WriteBufferManager::FreeMem(size_t mem) {
  int client_id = TG_GetThreadMetadata().client_id;
  if (client_id < 0) {
    std::cout << "[FAIRDB_LOG] Unaccounted FreeMem " << client_id << std::endl;
    return;
  }
  if (cache_res_mgr_ != nullptr) {
    FreeMemWithCache(mem);
  } else if (enabled()) {
    size_t current_memory = memory_used_.load(std::memory_order_relaxed);
    memory_used_.store(current_memory >= mem ? current_memory - mem : 0, std::memory_order_relaxed);

    size_t client_memory = per_client_memory_used_[client_id].load(std::memory_order_relaxed);
    per_client_memory_used_[client_id].store(client_memory >= mem ? client_memory - mem : 0, std::memory_order_relaxed);
  }
  mt_log_file_ << "wbm," << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << "," << client_id << ",free," << mem << "," << per_client_memory_used_[client_id].load(std::memory_order_relaxed) << std::endl;

  // Check if stall is active and can be ended.
  MaybeEndWriteStall();
}

void WriteBufferManager::FreeMemWithCache(size_t mem) {
  assert(cache_res_mgr_ != nullptr);
  std::lock_guard<std::mutex> lock(cache_res_mgr_mu_);

  size_t current_memory = memory_usage();
  size_t new_mem_used = current_memory >= mem ? current_memory - mem : 0;
  memory_used_.store(new_mem_used, std::memory_order_relaxed);

  Status s = cache_res_mgr_->UpdateCacheReservation(new_mem_used);

  // We absorb the error since WriteBufferManager is not able to handle
  // this failure properly.
  s.PermitUncheckedError();
}

void WriteBufferManager::BeginWriteStall(StallInterface* wbm_stall, int client_id) {
  assert(wbm_stall != nullptr);
  if (client_id < 0) {
    std::cout << "[FAIRDB_LOG] Unaccounted BeginWriteStall " << client_id << std::endl;
    return;
  }
  // Allocate outside of the lock.

  // Increment the stall count for the client atomically
  per_client_stall_count_[client_id].fetch_add(1, std::memory_order_relaxed);
  size_t current_total_stalls = total_stall_count_.fetch_add(1, std::memory_order_relaxed) + 1;

  // Print stall counts every 100 total stalls
  if (current_total_stalls % 1000 == 0) {
    std::cout << "Stall counts:" << std::endl;
    for (size_t i = 0; i < per_client_stall_count_.size(); ++i) {
      std::cout << "Client " << i << ": " << per_client_stall_count_[i].load(std::memory_order_relaxed) << " stalls" << std::endl;
    }
  }

  std::list<StallInterface*> new_node = {wbm_stall};
  {
    std::unique_lock<std::mutex> lock(mu_);
    // Verify if the stall conditions are still active.
    if (ShouldStall(client_id)) {
      per_client_stall_active_[client_id].store(true, std::memory_order_relaxed);
      per_client_queue_[client_id].splice(per_client_queue_[client_id].end(), std::move(new_node));
    }
  }

  // If the node was not consumed, the stall has ended already and we can signal
  // the caller.
  if (!new_node.empty()) {
    new_node.front()->Signal();
  }
}

// Called when memory is freed in FreeMem or the buffer size has changed.
void WriteBufferManager::MaybeEndWriteStall() {
  // Perform all deallocations outside of the lock.
  std::vector<std::list<StallInterface*>> cleanup(per_client_queue_.size());

  std::unique_lock<std::mutex> lock(mu_);
  for (size_t client_id = 0; client_id < per_client_queue_.size(); ++client_id) {
    if (!per_client_stall_active_[client_id].load(std::memory_order_relaxed)) {
      continue;  // Nothing to do for this client.
    }

    // Stall conditions have not been resolved for this client.
    if (allow_stall_.load(std::memory_order_relaxed) &&
        IsStallThresholdExceeded(client_id)) {
      continue;
    }

    // Unblock new writers for this client.
    per_client_stall_active_[client_id].store(false, std::memory_order_relaxed);

    // Unblock the writers in the queue.
    for (StallInterface* wbm_stall : per_client_queue_[client_id]) {
      wbm_stall->Signal();
    }
    cleanup[client_id] = std::move(per_client_queue_[client_id]);
  }
}

void WriteBufferManager::RemoveDBFromQueue(StallInterface* wbm_stall, int client_id) {
  assert(wbm_stall != nullptr);

  // Deallocate the removed nodes outside of the lock.
  std::list<StallInterface*> cleanup;

  if (enabled() && allow_stall_.load(std::memory_order_relaxed)) {
    std::unique_lock<std::mutex> lock(mu_);
    for (auto it = per_client_queue_[client_id].begin(); it != per_client_queue_[client_id].end();) {
      auto next = std::next(it);
      if (*it == wbm_stall) {
        cleanup.splice(cleanup.end(), per_client_queue_[client_id], std::move(it));
      }
      it = next;
    }
  }
  wbm_stall->Signal();
}

}  // namespace ROCKSDB_NAMESPACE
