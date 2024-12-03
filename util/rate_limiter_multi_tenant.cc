//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <algorithm>

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

#include "monitoring/statistics_impl.h"
#include "port/port.h"
#include "rocksdb/system_clock.h"
#include "test_util/sync_point.h"
#include "util/aligned_buffer.h"
#include "util/rate_limiter_multi_tenant_impl.h"
#include "rocksdb/tg_thread_local.h"

namespace ROCKSDB_NAMESPACE {

size_t RateLimiter::RequestToken(size_t bytes, size_t alignment,
                                 Env::IOPriority io_priority, Statistics* stats,
                                 RateLimiter::OpType op_type) {
  if (io_priority < Env::IO_TOTAL && IsRateLimited(op_type)) {
    bytes = std::min(bytes, static_cast<size_t>(GetSingleBurstBytes(op_type)));

    if (alignment > 0) {
      // Here we may actually require more than burst and block
      // as we can not write/read less than one page at a time on direct I/O
      // thus we do not want to be strictly constrained by burst
      bytes = std::max(alignment, TruncateToPageBoundary(alignment, bytes));
    }
    Request(bytes, io_priority, stats, op_type);
  }
  return bytes;
}

// Pending request.
struct MultiTenantRateLimiter::Req {
  explicit Req(int64_t _bytes, port::Mutex* _mu)
      : request_bytes(_bytes), bytes(_bytes), cv(_mu) {}
  int64_t request_bytes;
  int64_t bytes;
  port::CondVar cv;
};

// Key for map tracking pending requests.
struct MultiTenantRateLimiter::ReqKey {
  int client_id;
  Env::IOPriority priority;

  // Constructor with initializer list using this-> to avoid shadowing warning
  ReqKey(int c_id, Env::IOPriority pri) 
    : client_id(c_id), priority(pri) {}

  bool operator<(const ReqKey& other) const {
    if (client_id != other.client_id) {
      return client_id < other.client_id;
    }
    return priority < other.priority;
  }
};

MultiTenantRateLimiter::MultiTenantRateLimiter(
  int num_clients, std::vector<int64_t> bytes_per_sec, 
  std::vector<int64_t> read_bytes_per_sec, int64_t refill_period_us,
  int32_t fairness, RateLimiter::Mode mode, const std::shared_ptr<SystemClock>& clock, 
  int64_t single_burst_bytes)
    : RateLimiter(mode),
      refill_period_us_(refill_period_us),
      raw_single_burst_bytes_(single_burst_bytes),
      clock_(clock),
      stop_(false),
      exit_cv_(&request_mutex_),
      requests_to_wait_(0),
      next_refill_us_(NowMicrosMonotonicLocked()),
      fairness_(fairness > 100 ? 100 : fairness),
      rnd_((uint32_t)time(nullptr)),
      wait_until_refill_pending_(false),
      num_clients_(num_clients),
      rate_bytes_per_sec_(num_clients),
      refill_bytes_per_period_(num_clients),
      available_bytes_(num_clients),
      mode_(mode) {
  // Initialize per-priority counters.
  for (int i = Env::IO_LOW; i < Env::IO_TOTAL; ++i) {
    total_requests_[i] = 0;
    total_bytes_through_[i] = 0;
  }
  // Initialize per-client data structures.
  for (int i = 0; i < num_clients; ++i) {
    rate_bytes_per_sec_[i].store(bytes_per_sec[i]);
    refill_bytes_per_period_[i].store(CalculateRefillBytesPerPeriodLocked(bytes_per_sec[i]));
    available_bytes_[i].store(refill_bytes_per_period_[i].load(std::memory_order_relaxed));
    calls_per_client_.emplace_back(0);
    bytes_per_client_.emplace_back(0);
  }
  // Create (empty) queue for each client-priority pair.
  for (int c_id = 0; c_id < num_clients; ++c_id) {
    for (int pri = Env::IO_LOW; pri < Env::IO_TOTAL; ++pri) {
      ReqKey key(c_id, static_cast<Env::IOPriority>(pri));
      request_queue_map_[key] = std::deque<Req*>();
    }
  }

  std::cout << "[FAIRDB_LOG] per-client bytes/s limit:" << std::endl;
  for (int i = 0; i < num_clients_; ++i) {
    std::cout << "[FAIRDB_LOG] Client " << i << ": " << rate_bytes_per_sec_[i].load(std::memory_order_relaxed) << std::endl;
  }

  // TODO(tgriggs): create a separate (read) rate limiter
  if (read_bytes_per_sec.size() > 0) {
    read_rate_limiter_ = NewMultiTenantRateLimiter(
        num_clients,
        read_bytes_per_sec,  // <rate_limit> MB/s rate limit
        /* read_rate_limit = */ {},  // Hacky: don't create another rate limiter.
        refill_period_us,        // Refill period
        fairness,                // Fairness (default)
        rocksdb::RateLimiter::Mode::kReadsOnly, // Apply only to reads
        single_burst_bytes
    );
  }
}

MultiTenantRateLimiter::~MultiTenantRateLimiter() {
  MutexLock g(&request_mutex_);
  stop_ = true;
  std::deque<Req*>::size_type queues_size_sum = 0;
  for (const auto& pair : request_queue_map_) {
    queues_size_sum += pair.second.size();
  }
  requests_to_wait_ = static_cast<int32_t>(queues_size_sum);
  for (const auto& pair : request_queue_map_) {
    std::deque<Req*> queue = pair.second;
    for (auto& r : queue) {
      r->cv.Signal();
    }
  }
  while (requests_to_wait_ > 0) {
    exit_cv_.Wait();
  }
}

// This API allows user to dynamically change rate limiter's bytes per second.
void MultiTenantRateLimiter::SetBytesPerSecond(std::vector<int64_t> bytes_per_second) {
  MutexLock g(&request_mutex_);
  SetBytesPerSecondLocked(bytes_per_second);
}

void MultiTenantRateLimiter::SetBytesPerSecondLocked(std::vector<int64_t> bytes_per_second) {
  // TODO(tgriggs): Make this assert a debug statement.
  // assert(bytes_per_second > 0);
  for (size_t i = 0; i < bytes_per_second.size(); ++i) {
    rate_bytes_per_sec_[i].store(bytes_per_second[i], std::memory_order_relaxed);
    refill_bytes_per_period_[i].store(
        CalculateRefillBytesPerPeriodLocked(bytes_per_second[i]),
        std::memory_order_relaxed);
  }
}

void MultiTenantRateLimiter::SetBytesPerSecond(int64_t bytes_per_second) {
  (void) bytes_per_second;
  std::cout << "[FAIRDB_LOG] Deprecated 'SetBytesPerSecond'.\n";
}

void MultiTenantRateLimiter::SetBytesPerSecondLocked(int64_t bytes_per_second) {
  (void) bytes_per_second;
  std::cout << "[FAIRDB_LOG] Deprecated 'SetBytesPerSecondLocked'.\n";
}

int64_t MultiTenantRateLimiter::GetSingleBurstBytes() const {
  return GetSingleBurstBytes(TG_GetThreadMetadata().client_id);
}

int64_t MultiTenantRateLimiter::GetSingleBurstBytes(OpType op_type) const {
  if (op_type == RateLimiter::OpType::kRead) {
    if (read_rate_limiter_ != nullptr) {
      return read_rate_limiter_->GetSingleBurstBytes();
    } else {
      return 0;
    }
  }
  return GetSingleBurstBytes();
}

int64_t MultiTenantRateLimiter::GetSingleBurstBytes(int client_id) const {
  return refill_bytes_per_period_[client_id].load(std::memory_order_relaxed);
  // return 2 * 1024 * 1024;  // 2 MB

  // int64_t raw_single_burst_bytes =
  //     raw_single_burst_bytes_.load(std::memory_order_relaxed);
  // if (raw_single_burst_bytes == 0) {
  //   return refill_bytes_per_period_[client_id].load(std::memory_order_relaxed);
  // }
  // return raw_single_burst_bytes;
}

Status MultiTenantRateLimiter::SetSingleBurstBytes(int64_t single_burst_bytes) {
  if (single_burst_bytes < 0) {
    return Status::InvalidArgument(
        "`single_burst_bytes` must be greater than or equal to 0");
  }

  MutexLock g(&request_mutex_);
  raw_single_burst_bytes_.store(single_burst_bytes, std::memory_order_relaxed);
  return Status::OK();
}

// TODO(tgriggs): Used in tests. Do we need per-tenant?
  Status MultiTenantRateLimiter::GetTotalPendingRequests(
      int64_t* total_pending_requests,
      const Env::IOPriority pri) const {
    assert(total_pending_requests != nullptr);
    MutexLock g(&request_mutex_);
    int64_t total_pending_requests_sum = 0;
    for (const auto& pair : request_queue_map_) {
      if (pri == Env::IO_TOTAL || pair.first.priority == pri) {
        total_pending_requests_sum += static_cast<int64_t>(pair.second.size());
      }
    }
    *total_pending_requests = total_pending_requests_sum;
    return Status::OK();
  }

void MultiTenantRateLimiter::TGprintStackTrace() {
  void *array[20];
  size_t size;
  char **strings;
  size_t i;

  size = backtrace(array, 20);
  strings = backtrace_symbols(array, size);

  printf("Obtained %zd stack frames.\n", size);

  for (i = 0; i < size; i++)
      printf("%s\n", strings[i]);

  free(strings);
}

void MultiTenantRateLimiter::Request(int64_t bytes, const Env::IOPriority pri,
                                 Statistics* stats, OpType op_type) {
  if (op_type == RateLimiter::OpType::kRead) {
    if (read_rate_limiter_ != nullptr) {
      read_rate_limiter_->Request(bytes, pri, stats);
    }
    return;
  } else {
    Request(bytes, pri, stats);
  }
}

// TODO(tgriggs): the common case only hits a single client's rate limiter without any
//                queuing. So, let's not take any locks on this common case.
void MultiTenantRateLimiter::Request(int64_t bytes, const Env::IOPriority pri,
                                 Statistics* stats) {
  // Extract client ID from thread-local metadata.
  int client_id = TG_GetThreadMetadata().client_id;

  if (client_id == -2) {
    compaction_calls_++;
    compaction_bytes_ += bytes;
    return;
  }
  if (client_id < 0) {
    // std::cout << "[FAIRDB_LOG] Call to Request() has error in client id assignment: " << client_id << std::endl;
    // TGprintStackTrace();
    unassigned_calls_++;
    unassigned_bytes_ += bytes;
    return;
  }
  
  calls_per_client_[client_id]++;
  bytes_per_client_[client_id] += bytes;
  // if (total_calls_++ >= 1'000'000) {
  //   total_calls_ = 0;
  //   std::cout << "[FAIRDB_LOG] RL calls and bytes per-client for ";
  //   if (mode_ == rocksdb::RateLimiter::Mode::kReadsOnly) {
  //     std::cout << "READ: ";
  //   } else {
  //     std::cout << "WRITE: ";
  //   }
  //   std::cout << std::endl;
  //   for (size_t i = 0; i < calls_per_client_.size(); ++i) {
  //     std::cout << "Client " << i << ": " << calls_per_client_[i] << " calls, " << (bytes_per_client_[i]/1024/1024) << " MB\n";
  //   }
  //   std::cout << "Unassigned: " << unassigned_calls_ << " calls, " << (unassigned_bytes_/1024/1024) << " MB.\n";
  //   std::cout << "Compaction: " << compaction_calls_ << " calls, " << (compaction_bytes_/1024/1024) << " MB.\n";
  // }

  assert(bytes <= GetSingleBurstBytes(client_id));
  bytes = std::max(static_cast<int64_t>(0), bytes);
  TEST_SYNC_POINT("MultiTenantRateLimiter::Request");
  TEST_SYNC_POINT_CALLBACK("MultiTenantRateLimiter::Request:1",
                           &rate_bytes_per_sec_);
  
  total_requests_[pri].fetch_add(1, std::memory_order_relaxed);

  // **Attempt to consume available bytes without holding the mutex**
  int64_t available = available_bytes_[client_id].load(std::memory_order_relaxed);
  if (available > 0) {
    int64_t bytes_through = std::min(available, bytes);
    // Atomically decrease available bytes, ensuring it doesn't go negative
    int64_t expected = available;
    while (!available_bytes_[client_id].compare_exchange_weak(
        expected, expected - bytes_through, std::memory_order_relaxed)) {
      // Retry if the value changed
      bytes_through = std::min(expected, bytes);
    }
    total_bytes_through_[pri].fetch_add(bytes_through, std::memory_order_relaxed);
    bytes -= bytes_through;
  }

  if (bytes == 0) {
    // All bytes granted; no need to wait
    return;
  }

  // Acquire the mutex after checking if bytes == 0
  MutexLock g_lock(&request_mutex_);

  if (stop_) {
    return;
  }

  // Request cannot be satisfied at this moment, enqueue.
  Req req(bytes, &request_mutex_);
  request_queue_map_[ReqKey(client_id, pri)].push_back(&req);
  TEST_SYNC_POINT_CALLBACK("MultiTenantRateLimiter::Request:PostEnqueueRequest",
                           &request_mutex_);
  // A thread representing a queued request coordinates with other such threads.
  // There are two main duties.
  //
  // (1) Waiting for the next refill time.
  // (2) Refilling the bytes and granting requests.
  do {
    int64_t time_until_refill_us = next_refill_us_ - NowMicrosMonotonicLocked();
    if (time_until_refill_us > 0) {
      if (wait_until_refill_pending_) {
        // Somebody is performing (1). Trust we'll be woken up when our request
        // is granted or we are needed for future duties.
        req.cv.Wait();
      } else {
        // Whichever thread reaches here first performs duty (1) as described
        // above.
        int64_t wait_until = clock_->NowMicros() + time_until_refill_us;
        RecordTick(stats, NUMBER_RATE_LIMITER_DRAINS);
        wait_until_refill_pending_ = true;
        clock_->TimedWait(&req.cv, std::chrono::microseconds(wait_until));
        TEST_SYNC_POINT_CALLBACK("MultiTenantRateLimiter::Request:PostTimedWait",
                                 &time_until_refill_us);
        wait_until_refill_pending_ = false;
      }
    } else {
      // Whichever thread reaches here first performs duty (2) as described
      // above.
      RefillBytesAndGrantRequestsLocked();
    }
    if (req.request_bytes == 0) {
      // If there is any remaining requests, make sure there exists at least
      // one candidate is awake for future duties by signaling a front request
      // of a queue.
      for (int c_id = 0; c_id < num_clients_; ++c_id) {
        for (int priority = Env::IO_TOTAL - 1; priority >= Env::IO_LOW; --priority) {
          auto& queue = request_queue_map_[ReqKey(c_id, static_cast<Env::IOPriority>(priority))];
          if (!queue.empty()) {
            queue.front()->cv.Signal();
            break;
          }
        }
      }
    }
    // Invariant: non-granted request is always in one queue, and granted
    // request is always in zero queues.
// #ifndef NDEBUG
//     int num_found = 0;
//     for (int i = Env::IO_LOW; i < Env::IO_TOTAL; ++i) {
//       if (std::find(queue_[i].begin(), queue_[i].end(), &r) !=
//           queue_[i].end()) {
//         ++num_found;
//       }
//     }
//     if (r.request_bytes == 0) {
//       assert(num_found == 0);
//     } else {
//       assert(num_found == 1);
//     }
// #endif  // NDEBUG
  } while (!stop_ && req.request_bytes > 0);

  if (stop_) {
    // It is now in the clean-up of ~MultiTenantRateLimiter().
    // Therefore any woken-up request will have come out of the loop and then
    // exit here. It might or might not have been satisfied.
    --requests_to_wait_;
    exit_cv_.Signal();
  }
}

// TODO(tgriggs): Currently not doing this. I'm using strict priority. 
std::vector<Env::IOPriority>
MultiTenantRateLimiter::GeneratePriorityIterationOrderLocked() {
  std::vector<Env::IOPriority> pri_iteration_order(Env::IO_TOTAL /* 4 */);
  // We make Env::IO_USER a superior priority by always iterating its queue
  // first
  pri_iteration_order[0] = Env::IO_USER;

  bool high_pri_iterated_after_mid_low_pri = rnd_.OneIn(fairness_);
  TEST_SYNC_POINT_CALLBACK(
      "MultiTenantRateLimiter::GeneratePriorityIterationOrderLocked::"
      "PostRandomOneInFairnessForHighPri",
      &high_pri_iterated_after_mid_low_pri);
  bool mid_pri_itereated_after_low_pri = rnd_.OneIn(fairness_);
  TEST_SYNC_POINT_CALLBACK(
      "MultiTenantRateLimiter::GeneratePriorityIterationOrderLocked::"
      "PostRandomOneInFairnessForMidPri",
      &mid_pri_itereated_after_low_pri);

  if (high_pri_iterated_after_mid_low_pri) {
    pri_iteration_order[3] = Env::IO_HIGH;
    pri_iteration_order[2] =
        mid_pri_itereated_after_low_pri ? Env::IO_MID : Env::IO_LOW;
    pri_iteration_order[1] =
        (pri_iteration_order[2] == Env::IO_MID) ? Env::IO_LOW : Env::IO_MID;
  } else {
    pri_iteration_order[1] = Env::IO_HIGH;
    pri_iteration_order[3] =
        mid_pri_itereated_after_low_pri ? Env::IO_MID : Env::IO_LOW;
    pri_iteration_order[2] =
        (pri_iteration_order[3] == Env::IO_MID) ? Env::IO_LOW : Env::IO_MID;
  }

  TEST_SYNC_POINT_CALLBACK(
      "MultiTenantRateLimiter::GeneratePriorityIterationOrderLocked::"
      "PreReturnPriIterationOrder",
      &pri_iteration_order);
  return pri_iteration_order;
}

void MultiTenantRateLimiter::RefillBytesAndGrantRequestsLocked() {
  TEST_SYNC_POINT_CALLBACK(
      "MultiTenantRateLimiter::RefillBytesAndGrantRequestsLocked", &request_mutex_);
  next_refill_us_ = NowMicrosMonotonicLocked() + refill_period_us_;

  std::vector<int64_t> refill_bytes_per_period;
  for (int i = 0; i < num_clients_; ++i) {
    refill_bytes_per_period.push_back(refill_bytes_per_period_[i].load(std::memory_order_relaxed));
  }

  for (int c_id = 0; c_id < num_clients_; ++c_id) {
    int64_t refreshed_bytes = available_bytes_[c_id] + refill_bytes_per_period[c_id];
    available_bytes_[c_id] = std::min(refreshed_bytes, GetSingleBurstBytes(c_id));
  }

  // 1) iterate through clients
  // 2) for each client, do strict priority order from IO_USER to IO_LOW
  for (int client_id = 0; client_id < num_clients_; ++client_id) {
    for (int priority = Env::IO_TOTAL - 1; priority >= Env::IO_LOW; --priority) {
      ReqKey req_key = ReqKey(client_id, static_cast<Env::IOPriority>(priority));
      // auto queue_obj = request_queue_map_[req_key];
      auto* queue = &request_queue_map_[req_key];
      while (!queue->empty()) {
        auto* next_req = queue->front();
        if (available_bytes_[client_id] < next_req->request_bytes) {
          // Grant partial request_bytes even if request is for more than
          // `available_bytes_`, which can happen in a few situations:
          //
          // - The available bytes were partially consumed by other request(s)
          // - The rate was dynamically reduced while requests were already
          //   enqueued
          // - The burst size was explicitly set to be larger than the refill size
          next_req->request_bytes -= available_bytes_[client_id];
          available_bytes_[client_id] = 0;
          break;
        }
        available_bytes_[client_id] -= next_req->request_bytes;
        next_req->request_bytes = 0;
        total_bytes_through_[priority] += next_req->bytes;
        queue->pop_front();

        // Quota granted, signal the thread to exit
        next_req->cv.Signal();
      }
    }
  }
}

int64_t MultiTenantRateLimiter::CalculateRefillBytesPerPeriodLocked(
    int64_t rate_bytes_per_sec) {
  if (std::numeric_limits<int64_t>::max() / rate_bytes_per_sec <
      refill_period_us_) {
    // Avoid unexpected result in the overflow case. The result now is still
    // inaccurate but is a number that is large enough.
    return std::numeric_limits<int64_t>::max() / kMicrosecondsPerSecond;
  } else {
    return rate_bytes_per_sec * refill_period_us_ / kMicrosecondsPerSecond;
  }
}

RateLimiter* NewMultiTenantRateLimiter(
    int num_clients /* = 1 */,
    std::vector<int64_t> bytes_per_sec,
    std::vector<int64_t> read_bytes_per_sec,
    int64_t refill_period_us /* = 100 * 1000 */,
    int32_t fairness /* = 10 */,
    RateLimiter::Mode mode /* = RateLimiter::Mode::kWritesOnly */,
    int64_t single_burst_bytes /* = 0 */) {
  for (const int64_t limit : bytes_per_sec) {
    assert(limit > 0);
  }
  assert(refill_period_us > 0);
  assert(fairness > 0);
  std::unique_ptr<RateLimiter> limiter(new MultiTenantRateLimiter(
      num_clients, bytes_per_sec, read_bytes_per_sec, refill_period_us, fairness, mode,
      SystemClock::Default(), single_burst_bytes));
  return limiter.release();
}

}  // namespace ROCKSDB_NAMESPACE
