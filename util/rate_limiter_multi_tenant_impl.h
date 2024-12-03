//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "port/port.h"
#include "rocksdb/env.h"
#include "rocksdb/rate_limiter.h"
#include "rocksdb/status.h"
#include "rocksdb/system_clock.h"
#include "util/mutexlock.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

class MultiTenantRateLimiter : public RateLimiter {
 public:
  MultiTenantRateLimiter(
      int num_clients, std::vector<int64_t> writes_bytes_per_sec,
      std::vector<int64_t> read_bytes_per_sec, int64_t refill_period_us,
      int32_t fairness, RateLimiter::Mode mode,
      const std::shared_ptr<SystemClock>& clock, int64_t single_burst_bytes);

  virtual ~MultiTenantRateLimiter();

  // Permits dynamically change rate limiter's bytes per second.
  void SetBytesPerSecond(int64_t bytes_per_second) override;
  void SetBytesPerSecond(std::vector<int64_t> bytes_per_second);

  Status SetSingleBurstBytes(int64_t single_burst_bytes) override;

  // Request for token to write bytes. If this request can not be satisfied,
  // the call is blocked. Caller is responsible to make sure
  // bytes <= GetSingleBurstBytes(). Negative bytes passed in will be
  // rounded up to 0.
  using RateLimiter::Request;
  void Request(const int64_t bytes, const Env::IOPriority pri,
               Statistics* stats) override;
  void Request(const int64_t bytes, const Env::IOPriority pri,
               Statistics* stats, OpType op_type) override;

  Status GetTotalPendingRequests(
      int64_t* total_pending_requests,
      const Env::IOPriority pri = Env::IO_TOTAL) const override;

  // TODO: Make this per-tenant
  int64_t GetSingleBurstBytes() const override;
  int64_t GetSingleBurstBytes(OpType op_type) const override;

  int64_t GetTotalBytesThrough(
      const Env::IOPriority pri = Env::IO_TOTAL) const override;

  int64_t GetTotalBytesThroughForClient(int client_id) const override;

  int64_t GetTotalRequests(
      const Env::IOPriority pri = Env::IO_TOTAL) const override;

  // TODO: Make this per-client? Maybe thread-level storage?
  // Only used in [rocksdb/db/db_impl/db_impl_open.cc] to sanitize options
  int64_t GetBytesPerSecond() const override {
    int client_id = 0;
    return GetBytesPerSecond(client_id);
  }

  int64_t GetBytesPerSecond(int client_id) const {
    return rate_bytes_per_sec_[client_id].load(std::memory_order_relaxed);
  }

  RateLimiter* GetReadRateLimiter() override {
    return read_rate_limiter_;
  }

  virtual void TEST_SetClock(std::shared_ptr<SystemClock> clock) {
    MutexLock g(&request_mutex_);
    clock_ = std::move(clock);
    // Update next refill times for all clients
    for (int i = 0; i < num_clients_; ++i) {
      next_refill_us_[i] = NowMicrosMonotonicLocked();
    }
  }

 private:
  int64_t GetSingleBurstBytes(int client_id) const;
  static constexpr int kMicrosecondsPerSecond = 1000000;
  void RefillBytesAndGrantRequestsLocked();
  void RefillBytesAndGrantRequestsForClientLocked(int client_id);
  std::vector<Env::IOPriority> GeneratePriorityIterationOrderLocked();
  int64_t CalculateRefillBytesPerPeriodLocked(int64_t rate_bytes_per_sec);
  Status TuneLocked();
  void SetBytesPerSecondLocked(int64_t bytes_per_second);
  void SetBytesPerSecondLocked(std::vector<int64_t> bytes_per_second);

  void TGprintStackTrace();

  uint64_t NowMicrosMonotonicLocked() {
    return clock_->NowNanos() / std::milli::den;
  }

  // This mutex guards all internal states except per-client states
  mutable port::Mutex request_mutex_;

  const int64_t refill_period_us_;

  // This value is validated but unsanitized (may be zero).
  std::atomic<int64_t> raw_single_burst_bytes_;
  std::shared_ptr<SystemClock> clock_;

  std::atomic<bool> stop_{false};
  port::CondVar exit_cv_;
  std::atomic<int32_t> requests_to_wait_{0};

  // Make total_requests_ and total_bytes_through_ atomic
  std::atomic<int64_t> total_requests_[Env::IO_TOTAL];
  std::atomic<int64_t> total_bytes_through_[Env::IO_TOTAL];

  int32_t fairness_;
  Random rnd_;

  struct Req;
  struct ReqKey;

  // Multi-tenant extensions
  int num_clients_;
  std::vector<std::atomic<int64_t>> rate_bytes_per_sec_;
  std::vector<std::atomic<int64_t>> refill_bytes_per_period_;

  // available_bytes_ is now a vector of atomic integers
  std::vector<std::atomic<int64_t>> available_bytes_;

  std::map<ReqKey, std::deque<Req*>> request_queue_map_;

  // Per-client mutexes for fine-grained locking
  std::vector<port::Mutex> client_mutexes_;

  // Per-client flags and variables
  std::vector<std::atomic<bool>> wait_until_refill_pending_;
  std::vector<int64_t> next_refill_us_;

  // Tracking metrics
  std::vector<int64_t> calls_per_client_;
  std::vector<int64_t> bytes_per_client_;
  std::atomic<int64_t> unassigned_calls_{0};
  std::atomic<int64_t> unassigned_bytes_{0};
  std::atomic<int64_t> compaction_calls_{0};
  std::atomic<int64_t> compaction_bytes_{0};
  int total_calls_;

  RateLimiter* read_rate_limiter_ = nullptr;
  RateLimiter::Mode mode_;
};

}  // namespace ROCKSDB_NAMESPACE
