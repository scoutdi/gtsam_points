// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <gtsam_points/ann/incremental_voxelmap.hpp>

#include <gtsam_points/ann/knn_result.hpp>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/types/frame_traits.hpp>
#include <numeric>
#include <optional>

namespace gtsam_points {

template <typename VoxelContents>
IncrementalVoxelMap<VoxelContents>::IncrementalVoxelMap(double leaf_size)
: leaf_size_(leaf_size),
  inv_leaf_size(1.0 / leaf_size),
  lru_horizon(10),
  lru_counter(0),
  max_num_voxels_(20000),
  offsets(neighbor_offsets(7)) {}

template <typename VoxelContents>
IncrementalVoxelMap<VoxelContents>::~IncrementalVoxelMap() {}

template <typename VoxelContents>
void IncrementalVoxelMap<VoxelContents>::clear() {
  lru_counter = 0;
  flat_voxels.clear();
  voxels.clear();
}

template <typename VoxelContents>
void IncrementalVoxelMap<VoxelContents>::insert(const PointCloud& points, bool limit_hit_increment) {
  // Bumped up front, not after the point loop, so it is stable for the whole scan: insert(),
  // and then decay()/line_decay(), all share the per-iteration hit-counter bookkeeping and must
  // agree on which iteration "this one" is. Only relative ordering matters for LRU eviction, so
  // shifting every value by one is immaterial there.
  lru_counter++;
  const uint32_t iteration = static_cast<uint32_t>(lru_counter);

  // No per-iteration reset pass: points carry the iteration in which they were last adjusted,
  // so a stale stamp simply compares unequal. Clearing a flag for every point in every voxel
  // here used to cost 3.97 ms of this function's 12.87 ms on an Orin at 22k voxels.

  // Insert points to the voxelmap
  for (size_t i = 0; i < points.size(); i++) {
    const Eigen::Vector3i coord = fast_floor(points.points[i] * inv_leaf_size).template head<3>();

    auto found = voxels.find(coord);
    if (found == voxels.end()) {
      auto voxel = std::make_shared<std::pair<VoxelInfo, VoxelContents>>(VoxelInfo(coord, lru_counter), VoxelContents());

      found = voxels.emplace_hint(found, coord, flat_voxels.size());
      flat_voxels.emplace_back(voxel);
    }

    auto& [info, voxel] = *flat_voxels[found->second];
    info.lru = lru_counter;
    voxel.add(voxel_setting, points, i, limit_hit_increment, iteration);
  }

  // Finalize voxel means and covs
  for (auto& voxel : flat_voxels) {
    voxel->second.finalize();
  }
  // Eviction: triggered either by exceeding the voxel cap, or by accumulating too many empty
  // voxels (created when decay/line_decay drained all points from a voxel). Empty voxels are
  // cheap individually but grow unboundedly with sporadic noise, so we sweep them when they
  // exceed a fraction of the total. Tier-1 (empty removal) does the actual cleanup in both
  // cases; tier-2 (LRU eviction) only runs if we're still over the cap afterwards.
  last_evicted_voxels_ = 0;
  size_t empty_voxel_count = 0;
  size_t total_points = 0;
  for (const auto& v : flat_voxels) {
    const size_t n = frame::size(v->second);
    total_points += n;
    if (n == 0) {
      ++empty_voxel_count;
    }
  }
  // Two independent caps. The voxel cap bounds spatial extent and memory; the point cap bounds
  // registration cost, which scales with points-per-voxel rather than with voxel count, so a map
  // can be well under the voxel cap while still getting steadily more expensive to search.
  const bool over_voxel_cap = max_num_voxels_ > 0 && flat_voxels.size() > max_num_voxels_;
  const bool over_point_cap = max_num_points_ > 0 && total_points > max_num_points_;
  const bool over_cap = over_voxel_cap || over_point_cap;
  // Threshold of 1/4: bounds wasted empty-voxel memory to ~33% over the live set, while
  // amortizing the rebuild cost (each rebuild reclaims at least 25% of the storage).
  const bool too_many_empty = empty_voxel_count > flat_voxels.size() / 4;
  if (over_cap || too_many_empty) {
    const size_t before_eviction = flat_voxels.size();

    // Swap the voxel at `i` with the last one and drop it, patching `voxels` for just the one
    // entry that moved instead of rebuilding the whole hash map afterward -- the old code paid an
    // O(map size) unordered_map rebuild on every eviction regardless of how many voxels it
    // actually dropped. This means eviction no longer preserves voxel order in `flat_voxels`, but
    // nothing relies on that: decay() only needs eventual round-robin coverage, knn_search()'s tie
    // -breaking is about point order within a voxel, and the old tier-2 LRU sort already reordered
    // everything on every run it triggered.
    const auto remove_at = [&](size_t i) {
      voxels.erase(flat_voxels[i]->first.coord);
      const size_t last = flat_voxels.size() - 1;
      if (i != last) {
        flat_voxels[i] = std::move(flat_voxels[last]);
        voxels[flat_voxels[i]->first.coord] = i;
      }
      flat_voxels.pop_back();
    };

    // Tier 1: Remove empty voxels (those with no points left after pruning)
    for (size_t i = 0; i < flat_voxels.size();) {
      if (frame::size(flat_voxels[i]->second) == 0) {
        remove_at(i);
      } else {
        ++i;
      }
    }

    // Tier 2: If still over either cap, evict oldest voxels by LRU age. Removing empty voxels in
    // tier 1 did not change the point total, so total_points is still valid here.
    const bool still_over_voxels = max_num_voxels_ > 0 && flat_voxels.size() > max_num_voxels_;
    const bool still_over_points = max_num_points_ > 0 && total_points > max_num_points_;
    if (still_over_voxels || still_over_points) {
      // Extract the oldest voxels one at a time from a min-heap over indices (ranked by lru),
      // rather than fully sorting all of flat_voxels by lru: eviction is typically a small
      // fraction of the map (e.g. steady-state operation right at the cap evicts just enough to
      // make room for the next scan), so ranking the entire map to find the few oldest is wasted
      // work. Heapify is still O(N), but each extraction is only O(log N), against O(N log N) to
      // rank everything up front.
      std::vector<size_t> heap(flat_voxels.size());
      std::iota(heap.begin(), heap.end(), 0);
      const auto older = [&](size_t a, size_t b) {
        return flat_voxels[a]->first.lru > flat_voxels[b]->first.lru;
      };
      std::make_heap(heap.begin(), heap.end(), older);
      size_t heap_end = heap.size();
      const auto pop_oldest = [&]() {
        std::pop_heap(heap.begin(), heap.begin() + heap_end, older);
        return heap[--heap_end];
      };

      size_t num_to_evict = still_over_voxels ? flat_voxels.size() - max_num_voxels_ : 0;
      std::vector<size_t> victims;
      victims.reserve(num_to_evict);
      for (size_t i = 0; i < num_to_evict; i++) {
        victims.push_back(pop_oldest());
      }

      // Then keep taking the oldest until the point total is under the low-water target -- not
      // merely under the cap, or the next insert re-triggers and eviction runs on nearly every
      // scan. Bounded by flat_voxels.size() - 1 so a small target cannot empty the map entirely.
      if (max_num_points_ > 0) {
        size_t remaining_points = total_points;
        for (const size_t idx : victims) {
          remaining_points -= frame::size(flat_voxels[idx]->second);
        }
        while (remaining_points > target_num_points_ && victims.size() + 1 < flat_voxels.size()) {
          const size_t idx = pop_oldest();
          remaining_points -= frame::size(flat_voxels[idx]->second);
          victims.push_back(idx);
        }
      }

      // Evict the chosen victims via swap-remove, by their original flat_voxels index, walked
      // descending: each removal swaps in the current last element, which is only guaranteed not
      // to be one of the remaining not-yet-removed victims if the highest victim index goes first.
      std::sort(victims.begin(), victims.end(), [](size_t a, size_t b) { return a > b; });
      for (const size_t i : victims) {
        remove_at(i);
      }
    }

    last_evicted_voxels_ = before_eviction - flat_voxels.size();
  }
}

template <typename VoxelContents>
void IncrementalVoxelMap<VoxelContents>::decay(size_t step, size_t offset){
  for (size_t i = offset; i < flat_voxels.size(); i+=step) {
    flat_voxels[i]->second.decay(voxel_setting, static_cast<uint32_t>(lru_counter));
    flat_voxels[i]->second.remove_dead_points();
  }
}

template <typename VoxelContents>
void IncrementalVoxelMap<VoxelContents>::line_decay(Eigen::Vector4d start, Eigen::Vector4d dir, double start_length, double length, double radius_sq) {
  double progress = start_length;
  std::optional<Eigen::Vector3i> last_coord;
  while(progress < length){
    Eigen::Vector4d current_point = start + dir * progress;
    const Eigen::Vector3i coord = fast_floor(current_point * inv_leaf_size).template head<3>();
    if(last_coord && *last_coord == coord){
      progress += leaf_size_ * 0.25;
      continue;
    }

    auto found = voxels.find(coord);
    if (found != voxels.end()) {
      auto& [info, voxel] = *flat_voxels[found->second];
      voxel.line_decay(voxel_setting, start, dir, start_length, length, radius_sq, static_cast<uint32_t>(lru_counter));
    }

    progress += leaf_size_; // Step size for line decay
  }
}

template <typename VoxelContents>
double IncrementalVoxelMap<VoxelContents>::ray_trace(Eigen::Vector4d start, Eigen::Vector4d dir, double max_range, double ray_radius) const{
  double progress = 0.0;
  std::optional<Eigen::Vector3i> last_coord;
  double ray_radius_sq = ray_radius * ray_radius;

  while(progress < max_range){
    Eigen::Vector4d current_point = start + dir * progress;
    const Eigen::Vector3i coord = fast_floor(current_point * inv_leaf_size).template head<3>();

    if(last_coord && *last_coord == coord){
      progress += leaf_size_ * 0.25;
      continue;
    }
    last_coord = coord;

    auto found = voxels.find(coord);
    if (found != voxels.end()) {
      const auto& [info, voxel] = *flat_voxels[found->second];
      double hit_progress = voxel.ray_trace(voxel_setting, start, dir, max_range, ray_radius_sq);
      if(std::isfinite(hit_progress)){
        return hit_progress;
      }
    }

    // Do slower progress then for line decay to reduce risk of skipping a thin obstacle.
    // We accept less precission for decay as it has to run often and decaying is not critical.
    progress += 0.5*leaf_size_;
  }
  return std::numeric_limits<double>::infinity();
}

template <typename VoxelContents>
size_t IncrementalVoxelMap<VoxelContents>::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist) const {
  const Eigen::Vector4d query = (Eigen::Vector4d() << pt[0], pt[1], pt[2], 1.0).finished();
  const Eigen::Vector3i center = fast_floor(query * inv_leaf_size).template head<3>();

  size_t voxel_index = 0;
  const auto index_transform = [&](const size_t point_index) { return calc_index(voxel_index, point_index); };

  KnnResult<-1, decltype(index_transform)> result(k_indices, k_sq_dists, k, index_transform, max_sq_dist);
  for (const auto& offset : offsets) {
    const Eigen::Vector3i coord = center + offset;
    const auto found = voxels.find(coord);
    if (found == voxels.end()) {
      continue;
    }

    voxel_index = found->second;
    const auto& voxel = flat_voxels[voxel_index]->second;
    voxel.knn_search(voxel_setting, query, result);
  }

  return result.num_found();
}

template <typename VoxelContents>
std::vector<Eigen::Vector3i> IncrementalVoxelMap<VoxelContents>::neighbor_offsets(const int neighbor_voxel_mode) const {
  switch (neighbor_voxel_mode) {
    case 1:
      return std::vector<Eigen::Vector3i>{Eigen::Vector3i(0, 0, 0)};
    case 7:
      return std::vector<Eigen::Vector3i>{
        Eigen::Vector3i(0, 0, 0),
        Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(-1, 0, 0),
        Eigen::Vector3i(0, 1, 0),
        Eigen::Vector3i(0, -1, 0),
        Eigen::Vector3i(0, 0, 1),
        Eigen::Vector3i(0, 0, -1)};
    case 19: {
      std::vector<Eigen::Vector3i> offsets;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          for (int k = -1; k <= 1; k++) {
            if (std::abs(i) == 1 && std::abs(j) == 1 && std::abs(k) == 1) {
              continue;
            }

            offsets.push_back(Eigen::Vector3i(i, j, k));
          }
        }
      }
      return offsets;
    }
    case 27: {
      std::vector<Eigen::Vector3i> offsets;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          for (int k = -1; k <= 1; k++) {
            offsets.push_back(Eigen::Vector3i(i, j, k));
          }
        }
      }
      return offsets;
    }

    default:
      std::cerr << "error: invalid neighbor voxel mode " << neighbor_voxel_mode << std::endl;
      std::cerr << "     : neighbor voxel mode must be 1, 7, 19, or 27" << std::endl;
      return std::vector<Eigen::Vector3i>();
  }
}

template <typename VoxelContents>
bool IncrementalVoxelMap<VoxelContents>::has_points() const {
  return flat_voxels.empty() ? false : frame::has_points(flat_voxels.front()->second);
}

template <typename VoxelContents>
bool IncrementalVoxelMap<VoxelContents>::has_normals() const {
  return flat_voxels.empty() ? false : frame::has_normals(flat_voxels.front()->second);
}

template <typename VoxelContents>
bool IncrementalVoxelMap<VoxelContents>::has_covs() const {
  return flat_voxels.empty() ? false : frame::has_covs(flat_voxels.front()->second);
}

template <typename VoxelContents>
bool IncrementalVoxelMap<VoxelContents>::has_intensities() const {
  return flat_voxels.empty() ? false : frame::has_intensities(flat_voxels.front()->second);
}

template <typename VoxelContents>
std::vector<Eigen::Vector4d> IncrementalVoxelMap<VoxelContents>::voxel_points() const {
  std::vector<Eigen::Vector4d> points;
  points.reserve(flat_voxels.size() * 10);
  visit_points([&](const auto& voxel, const int i) { points.emplace_back(frame::point(voxel, i)); });
  return points;
}

template <typename VoxelContents>
std::vector<Eigen::Vector4d> IncrementalVoxelMap<VoxelContents>::voxel_normals() const {
  std::vector<Eigen::Vector4d> normals;
  normals.reserve(flat_voxels.size() * 10);
  visit_points([&](const auto& voxel, const int i) { normals.emplace_back(frame::normal(voxel, i)); });
  return normals;
}

template <typename VoxelContents>
std::vector<Eigen::Matrix4d> IncrementalVoxelMap<VoxelContents>::voxel_covs() const {
  std::vector<Eigen::Matrix4d> covs;
  covs.reserve(flat_voxels.size() * 10);
  visit_points([&](const auto& voxel, const int i) { covs.emplace_back(frame::cov(voxel, i)); });
  return covs;
}

template <typename VoxelContents>
std::vector<double> IncrementalVoxelMap<VoxelContents>::voxel_intensities() const {
  std::vector<double> intensities;
  intensities.reserve(flat_voxels.size() * 10);
  visit_points([&](const auto& voxel, const int i) { intensities.emplace_back(frame::intensity(voxel, i)); });
  return intensities;
}

template <typename VoxelContents>
PointCloudCPU::Ptr IncrementalVoxelMap<VoxelContents>::voxel_data() const {
  auto frame = std::make_shared<PointCloudCPU>();
  frame->points_storage.reserve(flat_voxels.size() * 10);
  if (has_normals()) {
    frame->normals_storage.reserve(flat_voxels.size() * 10);
  }
  if (has_covs()) {
    frame->covs_storage.reserve(flat_voxels.size() * 10);
  }
  if (has_intensities()) {
    frame->intensities_storage.reserve(flat_voxels.size() * 10);
  }

  visit_points([&](const auto& voxel, const int i) {
    // Dont return invalid points
    size_t counter = frame::hit_counter(voxel, i);
    if(counter < voxel_setting.valid_registration_count)
      return;

    frame->counters_storage.emplace_back(counter);
    frame->points_storage.emplace_back(frame::point(voxel, i));
    if (frame::has_normals(voxel)) {
      frame->normals_storage.emplace_back(frame::normal(voxel, i));
    }
    if (frame::has_covs(voxel)) {
      frame->covs_storage.emplace_back(frame::cov(voxel, i));
    }
    if (frame::has_intensities(voxel)) {
      frame->intensities_storage.emplace_back(frame::intensity(voxel, i));
    }
  });

  frame->num_points = frame->points_storage.size();
  frame->points = frame->points_storage.data();
  frame->normals = frame->normals_storage.empty() ? nullptr : frame->normals_storage.data();
  frame->covs = frame->covs_storage.empty() ? nullptr : frame->covs_storage.data();
  frame->intensities = frame->intensities_storage.empty() ? nullptr : frame->intensities_storage.data();

  return frame;
}

template <typename VoxelContents>
PointCloudCPU::Ptr IncrementalVoxelMap<VoxelContents>::voxel_data(Eigen::Vector4d center, double radius) const{
  auto frame = std::make_shared<PointCloudCPU>();
  auto radius_w_margin = radius + leaf_size_ * std::sqrt(3.0) / 2.0; // Add half diagonal of voxel to radius
  auto estimated_num_voxels = static_cast<size_t>(4*M_PI * std::pow(radius_w_margin / leaf_size_, 2));
  frame->points_storage.reserve(estimated_num_voxels * voxel_setting.max_num_points_in_cell);

  if (has_normals()) {
    frame->normals_storage.reserve(estimated_num_voxels * voxel_setting.max_num_points_in_cell);
  }
  if (has_covs()) {
    frame->covs_storage.reserve(estimated_num_voxels * voxel_setting.max_num_points_in_cell);
  }
  if (has_intensities()) {
    frame->intensities_storage.reserve(estimated_num_voxels * voxel_setting.max_num_points_in_cell);
  }

  auto radius_sq = radius_w_margin * radius_w_margin;
  for (const auto& voxel : flat_voxels) {
    Eigen::Vector3i voxel_id = voxel->first.coord;
    Eigen::Vector3d voxel_center = (voxel_id.cast<double>() + Eigen::Vector3d(0.5, 0.5, 0.5)) * leaf_size_;
    if((voxel_center - center.head<3>()).squaredNorm() > radius_sq)
      continue;

    // Iterate over points in voxel
    for (int i = 0; i < frame::size(voxel->second); i++) {
        size_t counter = frame::hit_counter(voxel->second, i);
        if(counter < voxel_setting.valid_obstacle_count)
          continue;

        frame->counters_storage.emplace_back(counter);
        frame->points_storage.emplace_back(frame::point(voxel->second, i));
        if (frame::has_normals(voxel->second)) {
          frame->normals_storage.emplace_back(frame::normal(voxel->second, i));
        }
        if (frame::has_covs(voxel->second)) {
          frame->covs_storage.emplace_back(frame::cov(voxel->second, i));
        }
        if (frame::has_intensities(voxel->second)) {
          frame->intensities_storage.emplace_back(frame::intensity(voxel->second, i));
        }
    }
  }

  frame->num_points = frame->points_storage.size();
  frame->points = frame->points_storage.data();
  frame->normals = frame->normals_storage.empty() ? nullptr : frame->normals_storage.data();
  frame->covs = frame->covs_storage.empty() ? nullptr : frame->covs_storage.data();
  frame->intensities = frame->intensities_storage.empty() ? nullptr : frame->intensities_storage.data();

  return frame;
}

}  // namespace gtsam_points
