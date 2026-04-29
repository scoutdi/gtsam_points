// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <queue>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam_points/ann/knn_result.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/frame_traits.hpp>

namespace gtsam_points {

/// @brief Point container with a flat vector.
struct FlatContainer {
public:
  /// @brief FlatContainer setting.
  struct Setting {
    void set_min_dist_in_cell(double dist) {
      this->min_dist_in_cell = dist;
      this->min_sq_dist_in_cell = dist * dist;
    }
    void set_max_num_points_in_cell(size_t num_points) { this->max_num_points_in_cell = num_points; }

    double min_dist_in_cell = 0.1;
    double min_sq_dist_in_cell = 0.1 * 0.1;  ///< Minimum squared distance between points in a cell.
    size_t max_num_points_in_cell = 20;      ///< Maximum number of points in a cell.

    void set_hit_counter_params(uint8_t initial_counter, uint8_t decay_decrement, uint8_t hit_increment, double hit_radius, uint8_t decay_upper_limit, uint8_t max_counter, uint8_t ray_trace_decrement, uint8_t valid_obstacle_count, uint8_t valid_registration_count, uint8_t limited_hit_counter_max) {
      this->initial_counter = initial_counter;
      this->decay_decrement = decay_decrement;
      this->hit_increment = hit_increment;
      this->hit_radius_sq = hit_radius * hit_radius;
      this->decay_upper_limit = decay_upper_limit;
      this->max_counter = max_counter;
      this->ray_trace_decrement = ray_trace_decrement;
      this->valid_obstacle_count = valid_obstacle_count;
      this->valid_registration_count = valid_registration_count;
      this->limited_hit_counter_max = limited_hit_counter_max;
    }

    uint8_t initial_counter = 5;
    uint8_t decay_decrement = 1;
    uint8_t hit_increment = 1;
    double hit_radius_sq = 0.2 * 0.2;
    uint8_t decay_upper_limit = 10;
    uint8_t max_counter = 100;
    uint8_t ray_trace_decrement = 5;
    uint8_t valid_obstacle_count = 10;
    uint8_t valid_registration_count = 1;
    uint8_t limited_hit_counter_max = 10;
  };

  /// @brief Constructor.
  FlatContainer() { points.reserve(10); }

  /// @brief Number of points.
  size_t size() const { return points.size(); }

  /// @brief Add a point to the container.
  void add(const Setting& setting, const PointCloud& points, size_t i, bool limit_hit_increment) {
    bool found_duplicate = false;

    for(int j=0; j<this->points.size(); j++){
      auto distance_sq = (this->points[j] - points.points[i]).squaredNorm();
      if (distance_sq < setting.hit_radius_sq) {
        auto max_hit_counter = limit_hit_increment ? setting.limited_hit_counter_max : setting.max_counter;
        increment_counter(setting, j, max_hit_counter);
      }
      if(distance_sq < setting.min_sq_dist_in_cell){
        found_duplicate = true;
        break;
      }
    }

    if (this->points.size() >= setting.max_num_points_in_cell || found_duplicate) {
      return;
    }

    this->points.emplace_back(points.points[i]);
    this->hit_counter.emplace_back(setting.initial_counter);
    this->hit_counter_adjusted_this_iteration.emplace_back(false);
    if (points.normals) {
      this->normals.emplace_back(points.normals[i]);
    }
    if (points.covs) {
      this->covs.emplace_back(points.covs[i]);
    }
    if (points.intensities) {
      this->intensities.emplace_back(points.intensities[i]);
    }
  }

  void decay(const Setting& setting) {
    if(hit_counter.size() != points.size()){
      throw std::runtime_error("hit_counter size mismatch in FlatContainer::decay");
    }
    for(size_t i=0; i<hit_counter.size(); i++){
      if(hit_counter[i] < setting.decay_upper_limit){
        decrement_counter(setting, i, setting.decay_decrement);
      }
    }
  }

  /// @brief Remove points with hit_counter == 0 (dead points).
  void remove_dead_points() {
    // First copy forward all good points to take the space of the points with 0 counter
    // Then delete the end of the list now only consisting of copies of points that have been moved forward
    size_t write = 0;
    for (size_t read = 0; read < points.size(); read++) {
      if (hit_counter[read] == 0) continue;
      if (write != read) {
        points[write] = points[read];
        hit_counter[write] = hit_counter[read];
        hit_counter_adjusted_this_iteration[write] = hit_counter_adjusted_this_iteration[read];
        if (!normals.empty()) normals[write] = normals[read];
        if (!covs.empty()) covs[write] = covs[read];
        if (!intensities.empty()) intensities[write] = intensities[read];
      }
      write++;
    }
    points.resize(write);
    hit_counter.resize(write);
    hit_counter_adjusted_this_iteration.resize(write);
    if (!normals.empty()) normals.resize(write);
    if (!covs.empty()) covs.resize(write);
    if (!intensities.empty()) intensities.resize(write);
  }

  void initialize_iteration(){
    for(size_t i=0; i<hit_counter_adjusted_this_iteration.size(); i++){
      hit_counter_adjusted_this_iteration[i] = false;
    }
  }

  /// @brief Finalize the container (Nothing to do for FlatContainer).
  void finalize() {}

  /// @brief Find k nearest neighbors.
  /// @param pt           Query point
  /// @param result       Result
  template <typename Result>
  void knn_search(const Setting& setting, const Eigen::Vector4d& pt, Result& result) const {
    if (points.empty()) {
      return;
    }

    for (size_t i = 0; i < points.size(); i++) {
      if(hit_counter[i] < setting.valid_registration_count){
        continue; // Skip invalid points
      }
      const double sq_dist = (points[i] - pt).squaredNorm();
      result.push(i, sq_dist);
    }
  }

  /// @brief Decay points along a line (ray tracing).
  /// @param setting      Container setting
  /// @param start        Start point of the line
  /// @param dir          Direction of the line
  /// @param length       Length of the line
  /// @param radius_sq    Squared radius around line where points will be decayed
  void line_decay(const Setting& setting, const Eigen::Vector4d& start, const Eigen::Vector4d& dir, const double start_length, const double length, const double radius_sq) {
    for (size_t i = 0; i < points.size(); i++) {
      Eigen::Vector4d pt_to_start = points[i] - start;

      double along_line_progress = pt_to_start.dot(dir);
      if(along_line_progress < start_length || along_line_progress > (length-setting.min_dist_in_cell)){
        continue; // Point is outside the line segment
      }
      double across_line_distance_sq = (pt_to_start - along_line_progress * dir).squaredNorm();
      if (across_line_distance_sq < radius_sq) {
        decrement_counter(setting, i, setting.ray_trace_decrement);
      }
    }
  }

  double ray_trace(const Setting& setting, const Eigen::Vector4d& start, const Eigen::Vector4d& dir, double max_range, double ray_radius_sq) const {
    double closest_hit = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < points.size(); i++) {
      if(hit_counter[i] < setting.valid_obstacle_count){
        continue; // Skip invalid points
      }
      Eigen::Vector4d pt_to_start = points[i] - start;

      double along_line_progress = pt_to_start.dot(dir);
      if(along_line_progress < 0.0 || along_line_progress > max_range){
        continue; // Point is outside the line segment
      }
      double across_line_distance_sq = (pt_to_start - along_line_progress * dir).squaredNorm();
      if(across_line_distance_sq < ray_radius_sq){
        closest_hit = std::min(closest_hit, along_line_progress);
      }
    }
    return closest_hit;
  }

public:
  std::vector<Eigen::Vector4d> points;   ///< Points
  std::vector<Eigen::Vector4d> normals;  ///< Normals
  std::vector<Eigen::Matrix4d> covs;     ///< Covariances
  std::vector<double> intensities;       ///< Intensities
  std::vector<uint8_t> hit_counter;
  std::vector<bool> hit_counter_adjusted_this_iteration;

private:
  void increment_counter(const Setting& setting, size_t index, uint8_t max_counter) {
    if(hit_counter_adjusted_this_iteration[index])
      return; // Already adjusted this iteration, skip to prevent multiple increments

    if (hit_counter[index] + setting.hit_increment <= max_counter) {
      hit_counter[index] += setting.hit_increment;
    } else {
      hit_counter[index] = max_counter;
    }
    hit_counter_adjusted_this_iteration[index] = true;
  }

  void decrement_counter(const Setting& setting, size_t index, uint8_t decrement) {
    if(hit_counter_adjusted_this_iteration[index])
      return; // Already adjusted this iteration, skip to prevent multiple decrements, or decrement of a poit that was seen this iteration.

    if (hit_counter[index] > decrement) {
      hit_counter[index] -= decrement;
    } else {
      hit_counter[index] = 0;
    }
    hit_counter_adjusted_this_iteration[index] = true;
  }
};

namespace frame {

template <>
struct traits<FlatContainer> {
  static int size(const FlatContainer& frame) { return frame.size(); }

  static bool has_points(const FlatContainer& frame) { return !frame.points.empty(); }
  static bool has_normals(const FlatContainer& frame) { return !frame.normals.empty(); }
  static bool has_covs(const FlatContainer& frame) { return !frame.covs.empty(); }
  static bool has_intensities(const FlatContainer& frame) { return !frame.intensities.empty(); }

  static const Eigen::Vector4d& point(const FlatContainer& frame, size_t i) { return frame.points[i]; }
  static const Eigen::Vector4d& normal(const FlatContainer& frame, size_t i) { return frame.normals[i]; }
  static const Eigen::Matrix4d& cov(const FlatContainer& frame, size_t i) { return frame.covs[i]; }
  static double intensity(const FlatContainer& frame, size_t i) { return frame.intensities[i]; }
  static uint8_t hit_counter(const FlatContainer& frame, size_t i) { return frame.hit_counter[i]; }
};

}  // namespace frame

}  // namespace gtsam_points
