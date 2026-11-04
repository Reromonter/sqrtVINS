/*
 * Square-root VINS
 * Copyright (C) 2025 Patrick Geneva
 * Copyright (C) 2025 Yuxiang Peng
 * Copyright (C) 2025 Guoquan Huang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_MOTION_PRIOR_H
#define OV_CORE_MOTION_PRIOR_H

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include "cam/CamBase.h"

namespace ov_core {

/**
 * @brief Predicted motion of the cameras between two image times.
 *
 * Holds the IMU poses in the global frame at an old image (the last one the
 * tracker saw, or the last keyframe) and at a new one, the IMU to camera
 * extrinsics, and optionally the ground plane n_G . p = d_G (n_G unit, up).
 * The trackers use it to predict where an old feature lands in the new image:
 * a point with known 3D position is projected with the full relative pose,
 * any other point is intersected with the ground plane (the plane-induced
 * homography), and without a plane only the rotation is applied.
 */
struct MotionPrior {

  /// If this prior can be used at all
  bool valid = false;

  /// Image times of the old and the new image
  double timestamp_old = -1;
  double timestamp_new = -1;

  /// IMU poses in the global frame at the old and the new image time
  Eigen::Matrix3d R_GtoIold = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_IoldinG = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_GtoInew = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_InewinG = Eigen::Vector3d::Zero();

  /// Ground plane n_G . p = d_G in the global frame (n_G points up)
  bool has_plane = false;
  Eigen::Vector3d n_G = Eigen::Vector3d::UnitZ();
  double d_G = 0;

  /// IMU to camera extrinsics of every camera
  std::map<size_t, Eigen::Matrix3d> R_ItoC;
  std::map<size_t, Eigen::Vector3d> p_IinC;

  /// Camera models (at tracking resolution) all prior geometry uses. These
  /// are the calibrated ones: with online intrinsics calibration the shared
  /// models drift while walking, which breaks stereo triangulation.
  std::map<size_t, std::shared_ptr<CamBase>> calib;

  /// Camera pose (R_GtoC, p_CinG) at the old (false) or new (true) time
  bool camera_pose(size_t cam_id, bool at_new, Eigen::Matrix3d &R_GtoC,
                   Eigen::Vector3d &p_CinG) const {
    if (R_ItoC.find(cam_id) == R_ItoC.end())
      return false;
    const Eigen::Matrix3d &R_GtoI = at_new ? R_GtoInew : R_GtoIold;
    const Eigen::Vector3d &p_IinG = at_new ? p_InewinG : p_IoldinG;
    R_GtoC = R_ItoC.at(cam_id) * R_GtoI;
    p_CinG = p_IinG - R_GtoC.transpose() * p_IinC.at(cam_id);
    return true;
  }

  /// Ray of a normalized coordinate of a camera at the old or new time in G
  bool ray_in_global(size_t cam_id, bool at_new, const cv::Point2f &uv_norm,
                     Eigen::Vector3d &origin, Eigen::Vector3d &dir) const {
    Eigen::Matrix3d R_GtoC;
    if (!camera_pose(cam_id, at_new, R_GtoC, origin))
      return false;
    dir = R_GtoC.transpose() * Eigen::Vector3d(uv_norm.x, uv_norm.y, 1.0);
    dir.normalize();
    return true;
  }

  /// Intersects a ray with the ground plane, false if it misses it in front
  bool intersect_plane(const Eigen::Vector3d &origin,
                       const Eigen::Vector3d &dir, Eigen::Vector3d &p_FinG,
                       double max_range = 50.0) const {
    if (!has_plane)
      return false;
    double denom = n_G.dot(dir);
    if (std::abs(denom) < 1e-6)
      return false;
    double t = (d_G - n_G.dot(origin)) / denom;
    if (t <= 0.05 || t > max_range)
      return false;
    p_FinG = origin + t * dir;
    return true;
  }

  /// Normalized coordinate of a raw pixel with the prior's camera model
  cv::Point2f undistort(size_t cam_id, const cv::Point2f &uv) const {
    return calib.at(cam_id)->undistort(uv);
  }

  /**
   * @brief Predicts the new raw pixel of an old raw pixel of camera cam_id
   * @param uv_old Raw (distorted) pixel in the old image
   * @param p_FinG Optional known 3D position of the feature in the global
   * frame; else the ground plane, else rotation only is used
   * @param uv_new Predicted raw pixel in the new image
   * @return false if the point projects behind the new camera
   */
  bool predict(size_t cam_id, const cv::Point2f &uv_old,
               const Eigen::Vector3d *p_FinG, cv::Point2f &uv_new) const {
    Eigen::Matrix3d R_GtoCnew;
    Eigen::Vector3d p_CnewinG;
    if (!valid || calib.find(cam_id) == calib.end() ||
        !camera_pose(cam_id, true, R_GtoCnew, p_CnewinG))
      return false;
    Eigen::Vector3d p_FinCnew;
    Eigen::Vector3d p_F;
    if (p_FinG != nullptr) {
      p_FinCnew = R_GtoCnew * (*p_FinG - p_CnewinG);
    } else {
      Eigen::Vector3d origin, dir;
      if (!ray_in_global(cam_id, false, undistort(cam_id, uv_old), origin,
                         dir))
        return false;
      if (intersect_plane(origin, dir, p_F)) {
        p_FinCnew = R_GtoCnew * (p_F - p_CnewinG);
      } else {
        // a point at infinity only sees the rotation
        p_FinCnew = R_GtoCnew * dir;
      }
    }
    if (p_FinCnew(2) < 0.05)
      return false;
    uv_new = calib.at(cam_id)->distort(
        cv::Point2f((float)(p_FinCnew(0) / p_FinCnew(2)),
                    (float)(p_FinCnew(1) / p_FinCnew(2))));
    return std::isfinite(uv_new.x) && std::isfinite(uv_new.y);
  }

  /**
   * @brief Triangulates a stereo observation at the old or new time
   * @param cam0,cam1 Camera ids of the pair
   * @param uvn0,uvn1 Normalized coordinates in both cameras
   * @param p_FinG Triangulated point in the global frame
   * @param max_depth Points further than this from cam0 are rejected
   */
  bool triangulate_stereo(size_t cam0, size_t cam1, bool at_new,
                          const cv::Point2f &uvn0, const cv::Point2f &uvn1,
                          Eigen::Vector3d &p_FinG,
                          double max_depth = 10.0) const {
    Eigen::Vector3d o0, d0, o1, d1;
    if (!ray_in_global(cam0, at_new, uvn0, o0, d0) ||
        !ray_in_global(cam1, at_new, uvn1, o1, d1))
      return false;
    // midpoint of the closest points of both rays
    Eigen::Vector3d w = o0 - o1;
    double b = d0.dot(d1), d = d0.dot(w), e = d1.dot(w);
    double denom = 1.0 - b * b;
    if (denom < 1e-9)
      return false;
    double s = (b * e - d) / denom, t = (e - b * d) / denom;
    if (s <= 0.05 || t <= 0.05 || s > max_depth)
      return false;
    Eigen::Vector3d p0 = o0 + s * d0, p1 = o1 + t * d1;
    // reject pairs whose rays do not meet (more than ~1 cm apart per metre)
    if ((p0 - p1).norm() > 0.01 * std::max(1.0, s))
      return false;
    p_FinG = 0.5 * (p0 + p1);
    return true;
  }
};

/**
 * @brief RANSAC plane fit n . p = d through 3D points
 * @param points Points (any frame)
 * @param up Expected up direction in that frame (the normal is flipped to it)
 * @param max_angle_deg Max angle between the fitted normal and up
 * @param inlier_m Inlier distance to the plane
 * @param min_inliers Min number of inliers for success
 * @param n Fitted unit normal (pointing along up)
 * @param d Fitted offset
 * @param num_inliers Number of inliers of the fit
 */
inline bool fit_ground_plane(const std::vector<Eigen::Vector3d> &points,
                             const Eigen::Vector3d &up, double max_angle_deg,
                             double inlier_m, int min_inliers,
                             Eigen::Vector3d &n, double &d, int &num_inliers) {
  num_inliers = 0;
  if ((int)points.size() < std::max(3, min_inliers))
    return false;
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> pick(0, points.size() - 1);
  double cos_max = std::cos(max_angle_deg * M_PI / 180.0);
  std::vector<size_t> best;
  for (int it = 0; it < 200; it++) {
    size_t a = pick(rng), b = pick(rng), c = pick(rng);
    if (a == b || b == c || a == c)
      continue;
    Eigen::Vector3d nn = (points[b] - points[a]).cross(points[c] - points[a]);
    if (nn.norm() < 1e-9)
      continue;
    nn.normalize();
    if (nn.dot(up) < 0)
      nn = -nn;
    if (nn.dot(up) < cos_max)
      continue;
    double dd = nn.dot(points[a]);
    std::vector<size_t> inl;
    for (size_t i = 0; i < points.size(); i++) {
      if (std::abs(nn.dot(points[i]) - dd) < inlier_m)
        inl.push_back(i);
    }
    if (inl.size() > best.size())
      best = inl;
  }
  if ((int)best.size() < min_inliers)
    return false;
  // least squares refit on the inliers
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (size_t i : best)
    mean += points[i];
  mean /= (double)best.size();
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (size_t i : best)
    cov += (points[i] - mean) * (points[i] - mean).transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
  n = es.eigenvectors().col(0).normalized();
  if (n.dot(up) < 0)
    n = -n;
  if (n.dot(up) < cos_max)
    return false;
  d = n.dot(mean);
  num_inliers = (int)best.size();
  return true;
}

/**
 * @brief Rigidity check: pairwise distances of 3D points must stay the same
 *
 * A pair is consistent when its distance changes by less than
 * tol_abs + tol_rel * (mean depth of both points), which follows the stereo
 * depth error. The largest mutually consistent group is grown greedily from the
 * point with the most consistent partners; points outside it are outliers.
 *
 * @param p_old,p_new Same points in the old and new camera frame
 * @return Inlier flags
 */
inline std::vector<bool>
rigidity_inliers(const std::vector<Eigen::Vector3d> &p_old,
                 const std::vector<Eigen::Vector3d> &p_new, double tol_abs,
                 double tol_rel, double min_share = 0.5) {
  size_t n = p_old.size();
  std::vector<bool> inlier(n, false);
  if (n < 4)
    return inlier;
  std::vector<std::vector<char>> ok(n, std::vector<char>(n, 0));
  std::vector<int> score(n, 0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      double d_old = (p_old[i] - p_old[j]).norm();
      double d_new = (p_new[i] - p_new[j]).norm();
      double depth = 0.25 * (p_old[i].norm() + p_old[j].norm() +
                             p_new[i].norm() + p_new[j].norm());
      if (std::abs(d_old - d_new) < tol_abs + tol_rel * depth) {
        ok[i][j] = ok[j][i] = 1;
        score[i]++;
        score[j]++;
      }
    }
  }
  // grow the consistent group from the best seed, strongest candidates first
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; i++)
    order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return score[a] > score[b]; });
  std::vector<size_t> group = {order[0]};
  for (size_t k = 1; k < n; k++) {
    size_t c = order[k];
    int agree = 0;
    for (size_t g : group)
      agree += ok[c][g];
    if (agree >= min_share * (double)group.size())
      group.push_back(c);
  }
  if (group.size() < 4)
    return inlier;
  for (size_t g : group)
    inlier[g] = true;
  return inlier;
}

} // namespace ov_core

#endif // OV_CORE_MOTION_PRIOR_H
