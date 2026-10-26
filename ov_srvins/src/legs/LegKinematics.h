/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2026      David Maerki
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program. If not, see
 * <https://www.gnu.org/licenses/>.
 */

#ifndef OV_SRVINS_LEG_KINEMATICS_H
#define OV_SRVINS_LEG_KINEMATICS_H

#include <array>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace ov_srvins {

/**
 * @brief Foot positions and Jacobians of a quadruped with closed-loop legs.
 *
 * Each leg of the LunarLeaper is a delta-like parallel mechanism: three motors
 * (T, L, R) and passive bearings closed by two point (spherical) joints. The
 * mechanism is read from a flat YAML exported from the robot's USD asset by
 * `legged_odometry/tools/export_mechanism.py` (lunarleaper-state-estimation):
 * a joint tree in parents-first order, the point closures, which joints are
 * the motors, and per leg its closures, passive joints, TL-arm body and foot.
 *
 * For given motor angles the passive angles are found by Newton's method on
 * the closure gaps, stepping the motors from the last consistent pose in
 * increments of at most `max_step` so the solution stays on the physical
 * assembly branch. The foot Jacobian follows from the implicit function
 * theorem on the closures: J = F_a - F_p C_p^-1 C_a.
 *
 * Everything is in double precision and in the mechanism's body frame,
 * independent of the filter's DataType.
 */
class LegKinematics {

public:
  static constexpr int kLegs = 4;
  static constexpr int kMotors = 12;
  typedef Eigen::Vector3d V3;
  typedef Eigen::Matrix3d M3;

  struct Result {
    std::array<V3, kLegs> foot;      ///< foot sphere centres, body frame [m]
    std::array<M3, kLegs> jacobian;  ///< d foot / d (T, L, R) [m/rad]
    std::array<bool, kLegs> converged; ///< the leg's loops closed
  };

  /// Load a mechanism file; throws std::runtime_error if it is unusable
  explicit LegKinematics(const std::string &mechanism_path);

  /**
   * @brief Solve the passives and compute feet and Jacobians.
   * @param q_motors Motor angles in motor_names() order (LF, RF, LH, RH x T, L, R)
   * @param out Feet and Jacobians; a leg whose loops cannot close keeps its
   * last consistent passive angles and reports converged = false
   * @return True if all four legs converged
   */
  bool solve(const Eigen::Matrix<double, kMotors, 1> &q_motors, Result &out);

  /// Forget the warm start (back to the assembled zero configuration)
  void reset() { q_good_.setZero(); }

  const std::vector<std::string> &motor_names() const { return motor_names_; }
  double foot_radius() const { return foot_radius_; }

  /// Accepted closure gap [m], Newton iterations per step, largest motor step [rad]
  double tol = 1e-9;
  int max_iter = 12;
  double max_step = 0.1;

private:
  struct Poses {
    std::vector<M3> body_R;
    std::vector<V3> body_p;
    std::vector<V3> joint_axis; ///< world axis of each joint
    std::vector<V3> joint_p;    ///< world pivot of each joint
  };

  void forward(const Eigen::VectorXd &q, Poses &poses) const;

  /// Closure gaps of a leg (6) and their Jacobian w.r.t. every joint (6 x nj)
  void closure(int leg, const Poses &poses, Eigen::Matrix<double, 6, 1> &r,
               Eigen::MatrixXd *J) const;

  /// Newton on one leg's passives; returns its largest closure gap
  double newton(int leg, Eigen::VectorXd &q, Poses &poses) const;

  // joint tree, parents first
  int num_bodies_ = 0, num_joints_ = 0;
  std::vector<int> parent_, child_;
  std::vector<V3> axis_, p0_, p1inv_;
  std::vector<M3> R0_, R1inv_;
  std::vector<bool> revolute_;
  std::vector<std::vector<bool>> ancestor_; ///< [body][joint]: joint on root -> body

  // closures: point pA of body cA coincides with point pB of body cB
  std::vector<int> cA_, cB_;
  std::vector<V3> pA_, pB_;

  // legs
  std::vector<std::string> motor_names_;
  std::array<std::array<int, 3>, kLegs> motor_joint_;
  std::array<std::array<int, 2>, kLegs> leg_closures_;
  std::array<std::array<int, 6>, kLegs> leg_passives_;
  std::array<int, kLegs> tl_body_;
  std::array<V3, kLegs> foot_offset_;
  double foot_radius_ = 0.0;

  /// Last consistent angles of every joint (warm start / branch tracking)
  Eigen::VectorXd q_good_;
};

} // namespace ov_srvins

#endif // OV_SRVINS_LEG_KINEMATICS_H
