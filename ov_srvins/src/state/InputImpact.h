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

#ifndef OV_SRVINS_STATE_INPUT_IMPACT_H
#define OV_SRVINS_STATE_INPUT_IMPACT_H

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "State.h"

namespace ov_srvins {

/**
 * @brief Per-frame account of what each input (IMU propagation, legs, the
 * visual update, the sun sensor, ZUPT) did to the current IMU state.
 *
 * Each step is measured between two snapshots of the IMU block: its value and
 * its marginal covariance P (from all rows of U, so it also holds before the
 * QR that follows a propagation at full clone count). Per state group g the
 * information is I_g = tr(P_g^-1), with the orientation error rotated into G
 * so that yaw (about G z) and tilt (G x, y) separate. With the sun alignment
 * yaw psi in the state (calib_sun_align), yaw is the heading in ENU,
 * theta_Gz - psi: in the filter's own frame G the sun mostly informs psi, not
 * the IMU. The steps telescope:
 *
 *   I_g(posterior) = I_g(after propagation) + sum over updates of dI_g
 *
 * so dI_g / I_g(posterior) is the share of the frame's estimate each update
 * supplied and I_g(after propagation) / I_g(posterior) the share carried by
 * the IMU-propagated past. For a scalar group these are exactly KH and 1 - KH
 * of the Kalman update. Propagation itself has dI_g <= 0. A recovery
 * inflation inside an updater (sun yaw, leg velocity) shows as a negative dI_g
 * of that updater.
 *
 * Each update step also records the correction it applied: its Mahalanobis
 * length against the step's prior covariance, and the yaw, tilt, position and
 * velocity changes.
 */
class InputImpact {

public:
  /// State groups, in log order
  enum Group { YAW, TILT, POS_XY, POS_Z, VEL, BG, BA, NUM_GROUPS };

  typedef Eigen::Matrix<double, NUM_GROUPS, 1> GroupVec;
  /// IMU error state (theta_I, p, v, bg, ba) plus the sun alignment yaw
  typedef Eigen::Matrix<double, 16, 16> Mat16d;
  typedef Eigen::Matrix<double, 16, 1> Vec16d;

  /// IMU block at one instant
  struct Snapshot {
    Eigen::Matrix3d R_GtoI;
    Eigen::Vector3d p, v, bg, ba;
    /// G-to-ENU alignment yaw, if the state estimates it
    bool has_align = false;
    double psi_align = 0.0;
    /// Marginal covariance of the error state; row / column 15 is psi
    /// (zero without has_align)
    Mat16d P;
  };

  /// One input's effect between two snapshots
  struct Step {
    std::string source;
    /// The updater accepted its measurement (always true for propagation)
    bool accepted = true;
    /// Information change per group
    GroupVec d_info = GroupVec::Zero();
    /// Mahalanobis length of the correction against the prior (sigma)
    double mahalanobis = 0.0;
    /// Yaw about G z (deg, signed), tilt (deg), position (m), velocity (m/s)
    double d_yaw_deg = 0.0, d_tilt_deg = 0.0, d_pos = 0.0, d_vel = 0.0;
  };

  static Snapshot snapshot(const std::shared_ptr<State> &state);

  /// tr(P_g^-1) per group, orientation in G
  static GroupVec information(const Snapshot &s);

  /// Start a frame at the state's current (posterior) IMU block
  void begin(const std::shared_ptr<State> &state, double timestamp);

  /// Close the step `source` ran since the last mark; propagation steps
  /// (`is_update` false) record no correction
  void mark(const std::string &source, const std::shared_ptr<State> &state,
            bool accepted = true, bool is_update = true);

  /// The frame as one [IMPACT] log line
  std::string line() const;

  const std::vector<Step> &steps() const { return steps_; }
  const GroupVec &posterior() const { return info_last_; }

private:
  double timestamp_ = -1.0;
  Snapshot last_;
  GroupVec info_last_ = GroupVec::Zero();
  std::vector<Step> steps_;
};

} // namespace ov_srvins

#endif // OV_SRVINS_STATE_INPUT_IMPACT_H
