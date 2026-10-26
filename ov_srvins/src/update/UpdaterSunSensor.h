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

#ifndef OV_SRVINS_UPDATER_SUNSENSOR_H
#define OV_SRVINS_UPDATER_SUNSENSOR_H

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "UpdaterOptions.h"
#include "state/State.h"
#include "utils/DataType.h"
#include "utils/sensor_data.h"

namespace ov_srvins {

/**
 * @brief Corrects the global yaw of the filter using a two-axis sun sensor.
 *
 * VIO cannot observe the heading of its own world frame G: gravity pins down
 * roll and pitch, and nothing pins down the rotation about gravity. A sun
 * sensor supplies the missing second reference direction and helps to correct the yaw.
 */
class UpdaterSunSensor {

public:
  /**
   * @brief Default constructor for our sun sensor updater.
   * @param options Updater options (chi2 multiplier)
   * @param sigma_alpha Noise (rad, 1-sigma) on the sensor's alpha angle
   * @param sigma_beta Noise (rad, 1-sigma) on the sensor's beta angle
   * @param R_ItoS Fixed rotation from the IMU frame to the sun sensor frame
   * @param min_elevation_rad Reject readings with the sun below this elevation
   * @param min_gravity_angle_rad Reject readings whose sun direction is within
   * this angle of gravity (the near-zenith degeneracy)
   * @param init_align_from_first_reading Solve the alignment yaw outright from
   * the first usable reading instead of letting the filter walk in to it
   */
  UpdaterSunSensor(UpdaterOptions &options, DataType sigma_alpha,
                   DataType sigma_beta, const Mat3 &R_ItoS,
                   DataType min_elevation_rad, DataType min_gravity_angle_rad,
                   bool init_align_from_first_reading = true);

  /// Mean raw gyro reading over [t0, t1] (IMU clock); false if not covered
  using GyroSource = std::function<bool(double, double, Vec3 &)>;

  /**
   * @brief Pair readings with the state on the IMU clock and rotate the
   * chosen reading to the state time with the gyro.
   *
   * Off, a reading is paired with the camera-clock state time and used as is,
   * which at 40 Hz leaves up to half a period (plus the camera-IMU offset)
   * between the attitude the reading saw and the one it is compared with:
   * ~0.1-0.3 deg while turning, several times the sensor noise.
   */
  void set_time_align(bool enable, GyroSource gyro) {
    time_align_ = enable;
    gyro_ = std::move(gyro);
  }

  /**
   * @brief Down-weight readings that fail the chi2 gate instead of dropping
   * them, up to a hard limit.
   *
   * A failing reading gets its noise inflated until it sits on the gate, so
   * it still pulls the state, by a bounded amount. Only a residual above
   * @p max_residual_rad (either angle) is rejected outright.
   */
  void set_robust(bool enable, DataType max_residual_rad) {
    robust_ = enable;
    max_residual_rad_ = max_residual_rad;
  }

  /**
   * @brief Re-acquire after a run of inconsistent readings.
   *
   * VIO is overconfident about its global yaw: once the heading has drifted
   * past the gate, its yaw covariance never grows enough to admit a reading
   * again and the sun stays locked out for good. When every reading for
   * @p after_s seconds failed the gate (rejected or down-weighted), the global
   * yaw of the whole trajectory is inflated so the next reading can correct
   * it. Zero disables this.
   */
  void set_recovery(DataType after_s) { recover_after_s_ = after_s; }

  /// Residual (measured - predicted angles, rad) of the last reading that
  /// reached the gate, accepted or not
  Vec2 last_residual() const { return last_residual_; }

  /**
   * @brief Feed function for sun sensor data
   * @param message Contains our timestamp and the two sensor angles
   * @param oldest_time Time that we can discard measurements before
   */
  void feed_sun(const ov_core::SunSensorData &message, double oldest_time = -1);

  /**
   * @brief Try to correct the global yaw with the freshest sun reading.
   *
   * Does not propagate: it reads the IMU state as the filter currently has it
   * at @p timestamp, so the caller should only call this when the state is
   * already at that time.
   *
   * @param state State of the filter
   * @param timestamp Time the state is currently at
   * @param sun_azimuth_rad Ephemeris azimuth, clockwise from true north
   * @param sun_elevation_rad Ephemeris elevation above the horizon
   * @return True if a measurement was accepted and applied
   */
  bool try_update(std::shared_ptr<State> state, double timestamp,
                  DataType sun_azimuth_rad, DataType sun_elevation_rad);

protected:
  /**
   * @brief Discard any buffered readings older than the given time
   * @param oldest_time Time that we can discard measurements before
   * @note The caller must already hold sun_data_mtx_.
   */
  void clean_old_sun_measurements(double oldest_time);

  /// Options used during update (chi2 multiplier)
  UpdaterOptions options_;

  /// Noise (rad, 1-sigma) on each of the two sensor angles
  DataType sigma_alpha_ = 0.0016;
  DataType sigma_beta_ = 0.0016;

  /// Rotation from the IMU frame to the sun sensor frame
  Mat3 R_ItoS_ = Mat3::Identity();

  /// Elevation below which a reading is rejected (rad)
  DataType min_elevation_rad_ = 0.0;

  /// Angle to gravity below which a reading is rejected (rad)
  DataType min_gravity_angle_rad_ = 0.0;

  /// Whether the alignment yaw is solved outright from the first reading
  bool init_align_from_first_reading_ = true;

  /// Set once that one-shot initialization has happened
  bool align_initialized_ = false;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, DataType> chi_squared_table_;

  /// Our history of sun sensor messages (time, alpha, beta)
  std::vector<ov_core::SunSensorData> sun_data_;

  /// Readings arrive on the subscriber thread while try_update runs on the
  /// update thread, exactly as with the IMU buffer in Propagator.
  std::mutex sun_data_mtx_;

  /// Timestamp of the reading we last consumed. A sun sensor is typically
  /// slower than the camera, so without this the same reading would be folded
  /// in on several consecutive frames and the filter would grow overconfident.
  double last_used_timestamp_ = -1;

  /// Last rejection reason we printed, so a permanently bad geometry says so
  /// once instead of once per reading
  std::string last_reject_reason_;

  /// See set_time_align / set_robust / set_recovery
  bool time_align_ = false;
  GyroSource gyro_;
  bool robust_ = false;
  DataType max_residual_rad_ = 5.0 * M_PI / 180.0;
  DataType recover_after_s_ = 0.0;
  Vec2 last_residual_ = Vec2::Zero();

  /// State time of the first reading of the current run of readings that
  /// failed the gate (-1: the last reading passed)
  double inconsistent_since_ = -1;

  /// Periodic statistics, printed every kStatsPeriod of state time
  void count_and_report(double timestamp, int *counter);
  double stats_since_ = -1;
  int n_accepted_ = 0, n_downweighted_ = 0, n_rejected_ = 0, n_no_reading_ = 0,
      n_recoveries_ = 0;
};

} // namespace ov_srvins

#endif // OV_SRVINS_UPDATER_SUNSENSOR_H
