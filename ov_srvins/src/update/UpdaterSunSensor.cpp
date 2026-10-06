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

#include "UpdaterSunSensor.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Type.h"
#include "utils/DataType.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/math/distributions/chi_squared.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>

using namespace ov_core;
using namespace ov_type;
using namespace ov_srvins;

namespace {

/// A sun reading is paired with the state time it is closest to; anything
/// further away than this is treated as "no reading for this state"
constexpr double kMaxSunDt = 0.1;

/// Hard cap on how much history the buffer keeps
constexpr double kMaxBufferAge = 5.0;

/// Sun direction in the sensor frame -> the two sensor angles.
Vec2 unit_vector_to_angles(const Vec3 &s) {
  return Vec2(std::atan2(s(0), s(2)), std::atan2(s(1), s(2)));
}

/// The two sensor angles -> sun direction in the sensor frame.
Vec3 angles_to_unit_vector(DataType alpha, DataType beta) {
  Vec3 v(std::tan(alpha), std::tan(beta), 1.0);
  return v.normalized();
}

/// Derivative of `unit_vector_to_angles` at @p s (2x3).
Eigen::Matrix<DataType, 2, 3> d_angles_d_vector(const Vec3 &s) {
  Eigen::Matrix<DataType, 2, 3> J = Eigen::Matrix<DataType, 2, 3>::Zero();
  const DataType den_a = s(0) * s(0) + s(2) * s(2);
  const DataType den_b = s(1) * s(1) + s(2) * s(2);
  J(0, 0) = s(2) / den_a;
  J(0, 2) = -s(0) / den_a;
  J(1, 1) = s(2) / den_b;
  J(1, 2) = -s(1) / den_b;
  return J;
}

/// Wrap an angle difference into (-pi, pi].
DataType wrap_pi(DataType angle) {
  while (angle > M_PI)
    angle -= 2.0 * M_PI;
  while (angle <= -M_PI)
    angle += 2.0 * M_PI;
  return angle;
}

} // namespace

UpdaterSunSensor::UpdaterSunSensor(UpdaterOptions &options,
                                   DataType sigma_alpha, DataType sigma_beta,
                                   const Mat3 &R_ItoS,
                                   DataType min_elevation_rad,
                                   DataType min_gravity_angle_rad,
                                   bool init_align_from_first_reading)
    : options_(options), sigma_alpha_(sigma_alpha), sigma_beta_(sigma_beta),
      R_ItoS_(R_ItoS), min_elevation_rad_(min_elevation_rad),
      min_gravity_angle_rad_(min_gravity_angle_rad),
      init_align_from_first_reading_(init_align_from_first_reading) {

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 1000; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table_[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterSunSensor::feed_sun(const ov_core::SunSensorData &message,
                                double oldest_time) {

  if (!std::isfinite(message.alpha) || !std::isfinite(message.beta)) {
    PRINT_DEBUG(YELLOW "[SUN]: dropped a non-finite reading at %.6f\n" RESET,
                message.timestamp);
    return;
  }

  std::lock_guard<std::mutex> lck(sun_data_mtx_);
  sun_data_.emplace_back(message);

  // Two lower bounds, whichever discards more: anything older than the state
  // can never be paired with it again, and nothing is kept past the cap.
  double retain_from = message.timestamp - kMaxBufferAge;
  if (oldest_time >= 0) {
    retain_from = std::max(retain_from, oldest_time - 0.10);
  }
  clean_old_sun_measurements(retain_from);
}

void UpdaterSunSensor::clean_old_sun_measurements(double oldest_time) {
  if (oldest_time < 0)
    return;
  auto it0 = sun_data_.begin();
  while (it0 != sun_data_.end()) {
    if (it0->timestamp < oldest_time) {
      it0 = sun_data_.erase(it0);
    } else {
      it0++;
    }
  }
}

bool UpdaterSunSensor::try_update(std::shared_ptr<State> state,
                                  double timestamp, DataType sun_azimuth_rad,
                                  DataType sun_elevation_rad) {

  // Nothing to do without a reading
  if (sun_data_.empty())
    return false;

  //=========================================================================
  // Geometry gating
  //=========================================================================
 
  const DataType max_elevation_rad = M_PI_2 - min_gravity_angle_rad_;
  if (sun_elevation_rad < min_elevation_rad_) {
    const std::string reason = "sun below the minimum elevation";
    if (last_reject_reason_ != reason) {
      last_reject_reason_ = reason;
      PRINT_INFO(YELLOW
                 "[SUN]: rejecting readings - elevation %.2f deg < %.2f deg\n"
                 RESET,
                 sun_elevation_rad * 180.0 / M_PI,
                 min_elevation_rad_ * 180.0 / M_PI);
    }
    return false;
  }
  if (sun_elevation_rad > max_elevation_rad) {
    const std::string reason = "sun too close to the zenith";
    if (last_reject_reason_ != reason) {
      last_reject_reason_ = reason;
      PRINT_INFO(YELLOW
                 "[SUN]: rejecting readings - elevation %.2f deg leaves only "
                 "%.2f deg to gravity (< %.2f deg), yaw is unobservable\n" RESET,
                 sun_elevation_rad * 180.0 / M_PI,
                 (M_PI_2 - sun_elevation_rad) * 180.0 / M_PI,
                 min_gravity_angle_rad_ * 180.0 / M_PI);
    }
    return false;
  }

  //=========================================================================
  // Pick the reading closest to the state time
  //=========================================================================
  ov_core::SunSensorData best;
  bool have_best = false;
  {
    std::lock_guard<std::mutex> lck(sun_data_mtx_);
    double best_dt = kMaxSunDt;
    for (const auto &meas : sun_data_) {
      if (meas.timestamp <= last_used_timestamp_)
        continue;
      const double dt = std::abs(meas.timestamp - timestamp);
      if (dt <= best_dt) {
        best_dt = dt;
        best = meas;
        have_best = true;
      }
    }
  }
  if (!have_best)
    return false;

  // Ephemeris direction in the gravity-aligned reference frame (ENU: x east,
  // y north, z up), with azimuth measured clockwise from true north.
  const Vec3 s_ENU(std::cos(sun_elevation_rad) * std::sin(sun_azimuth_rad),
                   std::cos(sun_elevation_rad) * std::cos(sun_azimuth_rad),
                   std::sin(sun_elevation_rad));

  //=========================================================================
  // One-shot initialization of the alignment yaw
  //=========================================================================
  // A linearized 2-DOF update cannot walk in from an arbitrary heading: the
  // chi2 gate sees the curvature of the measurement function as inconsistency
  // and rejects everything beyond a few degrees. So when the yaw starts out
  // genuinely unknown, solve it outright from the first reading.
  if (state->options.do_calib_sun_align && init_align_from_first_reading_ &&
      !align_initialized_) {
    const Vec3 s_meas = angles_to_unit_vector(best.alpha, best.beta);
    // The measured sun direction, carried back into G
    const Vec3 v = state->imu->Rot().transpose() * (R_ItoS_.transpose() * s_meas);
    // rot_z turns s_ENU's horizontal bearing into v's, so the yaw is just the
    // difference of the two bearings.
    const DataType psi_solved =
        wrap_pi(std::atan2(v(1), v(0)) - std::atan2(s_ENU(1), s_ENU(0)));

    VecX psi_vec(1);
    psi_vec(0) = psi_solved;
    state->calib_sun_align_yaw->set_value(psi_vec);
    state->calib_sun_align_yaw->set_fej(psi_vec);
    align_initialized_ = true;
    last_used_timestamp_ = best.timestamp;
    PRINT_INFO(CYAN "[SUN]: initialized the G-to-ENU yaw at %.4f deg from the "
                    "reading at %.6f (elevation mismatch %.3f deg)\n" RESET,
               psi_solved * 180.0 / M_PI, best.timestamp,
               (std::asin(std::max((DataType)-1.0,
                                   std::min((DataType)1.0, v(2)))) -
                sun_elevation_rad) *
                   180.0 / M_PI);
    return false;
  }

  //=========================================================================
  // Predicted measurement, at the *current* estimate
  //=========================================================================
  const DataType psi_align = state->calib_sun_align_yaw->value()(0);
  const Vec3 u = ov_core::rot_z(psi_align) * s_ENU;             // sun in G
  const Vec3 w = state->imu->Rot() * u;                // sun in I
  const Vec3 s_pred = R_ItoS_ * w;                     // sun in S

  // The sensor only reports angles for a sun in front of it, and so does the
  // prediction: behind the boresight both atan2 branches flip and the residual
  // would be meaningless.
  if (s_pred(2) <= 0.0) {
    const std::string reason = "predicted sun behind the sensor";
    if (last_reject_reason_ != reason) {
      last_reject_reason_ = reason;
      PRINT_INFO(YELLOW "[SUN]: rejecting readings - the predicted sun is "
                        "behind the sensor (s_z = %.3f)\n" RESET,
                 s_pred(2));
    }
    return false;
  }

  const Vec2 angles_pred = unit_vector_to_angles(s_pred);
  Vec2 res;
  res(0) = wrap_pi(best.alpha - angles_pred(0));
  res(1) = wrap_pi(best.beta - angles_pred(1));

  //=========================================================================
  // Jacobians, at the first estimates
  //=========================================================================
  // Exactly as the feature Jacobians do it: the residual above uses the live
  // estimate, everything from here on is evaluated at the FEJ point.
  const bool use_fej = state->options.do_fej;
  const DataType psi_align_jacob =
      use_fej ? state->calib_sun_align_yaw->fej()(0) : psi_align;
  const Mat3 R_GtoI_jacob =
      use_fej ? state->imu->Rot_fej() : state->imu->Rot();
  const Vec3 u_jacob = ov_core::rot_z(psi_align_jacob) * s_ENU;
  const Vec3 w_jacob = R_GtoI_jacob * u_jacob;
  const Vec3 s_jacob = R_ItoS_ * w_jacob;

  const Eigen::Matrix<DataType, 2, 3> d_angles_d_s = d_angles_d_vector(s_jacob);

  // Order of our Jacobian. These two are the only state elements the sun
  // touches: the measurement is a pure direction, so it has no position, no
  // velocity and no clone in it. Adding anything else here would let a yaw
  // correction leak into states the sun says nothing about.
  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(state->imu->q());
  const bool calibrating_align = state->options.do_calib_sun_align;
  if (calibrating_align) {
    Hx_order.push_back(state->calib_sun_align_yaw);
  }

  const int h_size = calibrating_align ? 4 : 3;
  MatX H_x = MatX::Zero(2, h_size);

  // d(s_pred)/d(theta): with the JPL left-multiplicative error state
  // R = (I - skew(dtheta)) * R_hat, a rotated vector picks up +skew(R_hat * u).
  H_x.leftCols<3>().noalias() = d_angles_d_s * (R_ItoS_ * skew_x(w_jacob));

  // d(s_pred)/d(psi_align): the derivative of a rotation about the fixed
  // vertical axis, i.e. the cross product z x u.
  if (calibrating_align) {
    const Vec3 z_axis(0.0, 0.0, 1.0);
    H_x.rightCols<1>().noalias() =
        d_angles_d_s * (R_ItoS_ * R_GtoI_jacob * skew_x(z_axis) * u_jacob);
  }

  //=========================================================================
  // Chi2 gate and square-root update
  //=========================================================================
  MatX U_dense, U_tri;
  StateHelper::get_marginal_U_block(state, Hx_order, U_dense, U_tri);
  MatX HUT = MatX::Zero(H_x.rows(), U_dense.rows() + U_tri.rows());
  HUT.leftCols(U_dense.rows()).noalias() = H_x * U_dense.transpose();
  HUT.rightCols(U_tri.rows()).noalias() =
      H_x * U_tri.transpose().triangularView<Eigen::Lower>();

  const DataType sigma_alpha_sq = sigma_alpha_ * sigma_alpha_;
  const DataType sigma_beta_sq = sigma_beta_ * sigma_beta_;

  Mat2 S = HUT * HUT.transpose();
  S.diagonal() += Vec2(sigma_alpha_sq, sigma_beta_sq);
  const DataType chi2 = res.dot(S.llt().solve(res));
  const DataType chi2_check =
      options_.chi2_multipler * chi_squared_table_[(int)res.rows()];
  if (chi2 > chi2_check) {
    PRINT_DEBUG(YELLOW "[SUN]: rejected reading at %.6f (chi2 %.3f > %.3f, "
                       "res %.3f / %.3f deg)\n" RESET,
                best.timestamp, chi2, chi2_check, res(0) * 180.0 / M_PI,
                res(1) * 180.0 / M_PI);
    
    const std::string reason = "chi2";
    if (last_reject_reason_ != reason) {
      last_reject_reason_ = reason;
      PRINT_INFO(YELLOW "[SUN]: readings are failing the chi2 gate (%.3f > "
                        "%.3f, res %.3f / %.3f deg) - the measurement and the "
                        "configured ephemeris disagree by more than the noise "
                        "allows; suppressing until one is accepted\n" RESET,
                 chi2, chi2_check, res(0) * 180.0 / M_PI,
                 res(1) * 180.0 / M_PI);
    }
    last_used_timestamp_ = best.timestamp;
    return false;
  }

  // Scatter H^T * R^-1 * res into a full-state vector, by variable id
  VecX RHTr = VecX::Zero(state->get_state_size());
  const Vec2 R_inv_res(res(0) / sigma_alpha_sq, res(1) / sigma_beta_sq);
  int local_id = 0;
  for (const auto &var : Hx_order) {
    RHTr.middleRows(var->id(), var->size()).noalias() +=
        H_x.block(0, local_id, H_x.rows(), var->size()).transpose() * R_inv_res;
    local_id += var->size();
  }

  // Whiten: R^(-1/2) * H * U^T, one scale per measurement row
  HUT.row(0) /= sigma_alpha_;
  HUT.row(1) /= sigma_beta_;

  // Sun readings arrive on their own schedule rather than with the camera, so
  // this update stands on its own instead of joining the per-frame batch.
  state->setup_matrix_buffer();
  state->store_update_factor(HUT, RHTr);
  StateHelper::update_llt(state);
  state->clear();

  last_used_timestamp_ = best.timestamp;
  last_reject_reason_.clear();
  PRINT_INFO(CYAN "[SUN]: accepted reading at %.6f (chi2 %.3f < %.3f, res "
                  "%.3f / %.3f deg, psi_align %.3f deg)\n" RESET,
             best.timestamp, chi2, chi2_check, res(0) * 180.0 / M_PI,
             res(1) * 180.0 / M_PI,
             state->calib_sun_align_yaw->value()(0) * 180.0 / M_PI);
  return true;
}
