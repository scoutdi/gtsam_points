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

    uint8_t initial_counter = 5;
    uint8_t decay_decrement = 1;
    uint8_t hit_increment = 1;
    uint8_t decay_upper_limit = 10;
    uint8_t max_counter = 100;
    uint8_t ray_trace_decrement = 5;
    uint8_t valid_obstacle_count = 10;
  };

  /// @brief Constructor.
  FlatContainer() { points.reserve(10); }

  /// @brief Number of points.
  size_t size() const { return points.size(); }

  /// @brief Add a point to the container.
  void add(const Setting& setting, const PointCloud& points, size_t i) {
    bool found_duplicate = false;

    for(int j=0; j<this->points.size(); j++){
      auto distance_sq = (this->points[j] - points.points[i]).squaredNorm();
      if(distance_sq < setting.min_sq_dist_in_cell){
        found_duplicate = true;
        break;
      }
      if(distance_sq < setting.min_sq_dist_in_cell * 16){ // If the point is close enough, increment the hit_counter to keep it alive longer
        increment_counter(setting, j);
      }
    }

    if (this->points.size() >= setting.max_num_points_in_cell || found_duplicate) {
      return;
    }

    this->points.emplace_back(points.points[i]);
    this->hit_counter.emplace_back(setting.initial_counter);
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
      if(hit_counter[i] < 10){
        decrement_counter(setting, i, setting.decay_decrement);
      }
    }
  }

  /// @brief Finalize the container (Nothing to do for FlatContainer).
  void finalize() {}

  /// @brief Find k nearest neighbors.
  /// @param pt           Query point
  /// @param result       Result
  template <typename Result>
  void knn_search(const Eigen::Vector4d& pt, Result& result) const {
    if (points.empty()) {
      return;
    }

    for (size_t i = 0; i < points.size(); i++) {
      const double sq_dist = (points[i] - pt).squaredNorm();
      result.push(i, sq_dist);
    }
  }

  /// @brief Decay points along a line (ray tracing).
  /// @param setting      Container setting
  /// @param start        Start point of the line
  /// @param dir          Direction of the line
  /// @param length       Length of the line
  void line_decay(const Setting& setting, const Eigen::Vector4d& start, const Eigen::Vector4d& dir, const double length) {
    if (points.empty()) {
      return;
    }

    for (size_t i = 0; i < points.size(); i++) {
      Eigen::Vector4d pt_to_start = points[i] - start;

      double along_line_progress = pt_to_start.dot(dir);
      if(along_line_progress < 0.0 || along_line_progress > (length-setting.min_dist_in_cell)){ 
        continue; // Point is outside the line segment
      }
      double across_line_distance_sq = (pt_to_start - along_line_progress * dir).squaredNorm();
      if(across_line_distance_sq < setting.min_sq_dist_in_cell){
        decrement_counter(setting, i, setting.ray_trace_decrement);
      }
    }
  }

  double ray_trace(const Setting& setting, const Eigen::Vector4d& start, const Eigen::Vector4d& dir, double max_range, double ray_radius_sq) const {
    double closest_hit = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < points.size(); i++) {
      if(hit_counter[i] == 0){
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

private:
  void increment_counter(const Setting& setting, size_t index) {
    if (hit_counter[index] + setting.hit_increment <= setting.max_counter) {
      hit_counter[index] += setting.hit_increment;
    } else {
      hit_counter[index] = setting.max_counter;
    }
  }

  void decrement_counter(const Setting& setting, size_t index, uint8_t decrement) {
    if (hit_counter[index] > decrement) {
      hit_counter[index] -= decrement;
    } else {
      hit_counter[index] = 0;
    }
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
