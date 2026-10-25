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

#ifndef OV_SRVINS_UPDATER_LEGVELOCITY_H
#define OV_SRVINS_UPDATER_LEGVELOCITY_H

#include <array>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "UpdaterOptions.h"
#include "legs/LegKinematics.h"
#include "state/State.h"
#include "utils/DataType.h"
#include "utils/sensor_data.h"

namespace ov_srvins {

/**
 * @brief Velocity update from the legs of a legged robot.
 *
 * A foot in stance does not move in the world, so the IMU moves with the
 * negative of that foot's velocity relative to the IMU (Bloesch et al., IROS
 * 2013; Cerberus, ICRA 2023). With the foot in the IMU frame
 * f = R_BtoI foot_B + p_BinI, the motor rates dq and the raw gyro w_m:
 *
 *   m = -s R_BtoI J(q) dq + [f]x w_m       (joint data + gyro only)
 *   h = R_GtoI v_IinG + [f]x b_g           (from the state)
 *
 * with J the foot Jacobian of the closed-loop leg (LegKinematics) and s an
 * optional kinematic scale. Every stance leg gives one 3D residual m - h; each
 * passes its own chi2 test (a slipping or wrongly detected foot disagrees with
 * the state), and the accepted ones are applied together.
 *
 * Contact comes from the motor torques: the quasi-static foot force
 * F = -J^-T tau, its vertical part along the filter's gravity, and the share
 * of each foot in the summed vertical force with hysteresis (stance above
 * `share_on`, swing below `share_off`). A ratio, so neither the torque scale
 * nor the robot mass enter.
 *
 * Joint samples are averaged over a window around the state time, which
 * removes most of the stride-frequency noise of a single sample.
 */
class UpdaterLegVelocity {

public:
  struct Params {
    /// Rotation body -> IMU (v_I = R_BtoI v_B) and the body origin in the IMU
    Mat3 R_BtoI = Mat3::Identity();
    Vec3 p_BinI = Vec3::Zero();
    /// Noise of one leg's velocity over the window (m/s, 1-sigma per axis).
    /// 0.1 was the most accurate on BWL (bwl-II/III ATE 0.4-0.7 m vs 0.9-1.6
    /// at 0.2), but it relies on the lock-out recovery below: without it the
    /// gate locked the legs out for good at bwl-III's first visual break.
    DataType sigma_v = 0.1;
    /// Half width of the averaging window around the state time (s)
    double half_window_s = 0.1;
    /// t_joint = t_imu + time_offset_s
    double time_offset_s = 0.0;
    /// Load-share stance thresholds (on / off) and the share of the window a
    /// leg must be in stance to be used
    double share_on = 0.25;
    double share_off = 0.15;
    double min_stance_fraction = 0.8;
    /// Kinematic scale on the leg velocity (Cerberus-style leg calibration)
    DataType velocity_scale = 1.0;
    /// Skip the update when the newest joint sample is older than this (s)
    double max_age_s = 0.5;
    /// After this long with every stance leg failing the gate, inflate the
    /// velocity uncertainty so the legs can pull the state back (s, 0 = never)
    double recover_after_s = 1.0;
  };

  /// Mean raw gyro reading over [t0, t1] (IMU clock); false if not covered
  using GyroSource = std::function<bool(double, double, Vec3 &)>;

  UpdaterLegVelocity(UpdaterOptions &options, const Params &params,
                     std::shared_ptr<LegKinematics> kinematics,
                     GyroSource gyro);

  /**
   * @brief Feed function for joint states
   * @param message Timestamp (IMU clock) and the joints by name
   * @param oldest_time Time that we can discard measurements before
   */
  void feed_joint_state(const ov_core::JointStateData &message,
                        double oldest_time = -1);

  /**
   * @brief Apply the leg velocity update at the current state time.
   *
   * Does not propagate: the state must already be at @p timestamp.
   * @param state State of the filter
   * @param timestamp Time the state is at (camera clock)
   * @return True if at least one leg was applied
   */
  bool try_update(std::shared_ptr<State> state, double timestamp);

  /// Legs applied / rejected by the gate in the last call
  int last_accepted() const { return last_accepted_; }
  int last_rejected() const { return last_rejected_; }
  /// Residuals (m/s, IMU frame) of the last call's stance legs
  const std::vector<Vec3> &last_residuals() const { return last_residuals_; }

protected:
  /// One joint sample after kinematics and contact detection
  struct Sample {
    double timestamp;
    std::array<bool, LegKinematics::kLegs> stance;
    std::array<Vec3, LegKinematics::kLegs> foot_I;  ///< foot in the IMU frame
    std::array<Vec3, LegKinematics::kLegs> vrel_I;  ///< -R_BtoI J dq
  };

  /// Run kinematics + contact on the joint samples that arrived since last time
  void process_new_samples(const Vec3 &up_B);

  UpdaterOptions options_;
  Params params_;
  std::shared_ptr<LegKinematics> kin_;
  GyroSource gyro_;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, DataType> chi_squared_table_;

  /// Raw joint states, fed from the subscriber thread
  std::deque<ov_core::JointStateData> raw_;
  std::mutex raw_mtx_;

  /// Processed samples (update thread only)
  std::deque<Sample> samples_;
  std::array<bool, LegKinematics::kLegs> stance_state_{};

  /// message joint index of each motor, resolved from the first message
  std::vector<int> joint_index_;
  bool names_ok_ = false;

  int last_accepted_ = 0, last_rejected_ = 0;

  /// State time of the first update of the current run in which every stance
  /// leg failed the gate (-1: the last update accepted a leg)
  double inconsistent_since_ = -1;
  std::vector<Vec3> last_residuals_;

  /// Periodic statistics, printed every kStatsPeriod of state time
  void report(double timestamp);
  double stats_since_ = -1;
  int n_updates_ = 0, n_no_data_ = 0, n_legs_accepted_ = 0,
      n_legs_rejected_ = 0, n_fk_failed_ = 0, n_recoveries_ = 0;
};

} // namespace ov_srvins

#endif // OV_SRVINS_UPDATER_LEGVELOCITY_H
