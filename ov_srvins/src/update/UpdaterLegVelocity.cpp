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

#include "UpdaterLegVelocity.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Type.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/math/distributions/chi_squared.hpp>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>

using namespace ov_srvins;
using namespace ov_core;

namespace {

/// How often the statistics line is printed (s of state time)
constexpr double kStatsPeriod = 10.0;

/// Keep this much processed joint history behind the newest sample (s)
constexpr double kHistory = 2.0;

} // namespace

UpdaterLegVelocity::UpdaterLegVelocity(UpdaterOptions &options,
                                       const Params &params,
                                       std::shared_ptr<LegKinematics> kinematics,
                                       GyroSource gyro)
    : options_(options), params_(params), kin_(std::move(kinematics)),
      gyro_(std::move(gyro)) {
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table_[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
  stance_state_.fill(false);
}

void UpdaterLegVelocity::feed_joint_state(const JointStateData &message,
                                          double oldest_time) {
  std::lock_guard<std::mutex> lck(raw_mtx_);
  raw_.push_back(message);
  dbg_fed_++;
  // Nothing consumes the queue before the filter is initialized
  while (oldest_time > 0 && !raw_.empty() &&
         raw_.front().timestamp < oldest_time)
    raw_.pop_front();
}

void UpdaterLegVelocity::process_new_samples(const Vec3 &up_B) {
  std::deque<JointStateData> fresh;
  {
    std::lock_guard<std::mutex> lck(raw_mtx_);
    fresh.swap(raw_);
  }
  std::sort(fresh.begin(), fresh.end());

  const auto &motors = kin_->motor_names();
  const LegKinematics::M3 R_BtoI = params_.R_BtoI.cast<double>();
  const LegKinematics::V3 p_BinI = params_.p_BinI.cast<double>();
  const LegKinematics::V3 up = up_B.cast<double>();
  for (const auto &msg : fresh) {
    if (!samples_.empty() && msg.timestamp <= samples_.back().timestamp) {
      dbg_old_++;
      continue;
    }
    if (!names_ok_) {
      joint_index_.assign(motors.size(), -1);
      for (size_t m = 0; m < motors.size(); m++) {
        auto it = std::find(msg.names.begin(), msg.names.end(), motors[m]);
        if (it == msg.names.end()) {
          PRINT_WARNING(YELLOW "[LEG]: joint state has no %s - leg update "
                               "disabled until it does\n" RESET,
                        motors[m].c_str());
          joint_index_.clear();
          break;
        }
        joint_index_[m] = (int)(it - msg.names.begin());
      }
      names_ok_ = !joint_index_.empty();
      if (!names_ok_)
        continue;
    }
    const size_t n = msg.names.size();
    if (msg.position.size() != n || msg.velocity.size() != n ||
        msg.effort.size() != n) {
      dbg_bad_size_++;
      continue; // the leg update needs angles, rates and torques
    }
    dbg_processed_++;

    Eigen::Matrix<double, LegKinematics::kMotors, 1> q, dq, tau;
    for (int m = 0; m < LegKinematics::kMotors; m++) {
      q(m) = msg.position[joint_index_[m]];
      dq(m) = msg.velocity[joint_index_[m]];
      tau(m) = msg.effort[joint_index_[m]];
    }
    LegKinematics::Result kin;
    kin_->solve(q, kin);

    // quasi-static foot forces F = -J^-T tau, and the vertical load share
    std::array<double, LegKinematics::kLegs> fz{};
    double fz_sum = 0.0;
    for (int leg = 0; leg < LegKinematics::kLegs; leg++) {
      if (!kin.converged[leg])
        continue;
      const LegKinematics::V3 F =
          -kin.jacobian[leg].transpose().fullPivLu().solve(
              tau.segment<3>(3 * leg));
      fz[leg] = std::max(0.0, F.dot(up));
      fz_sum += fz[leg];
    }

    Sample s;
    s.timestamp = msg.timestamp;
    for (int leg = 0; leg < LegKinematics::kLegs; leg++) {
      const double share = fz_sum > 1e-9 ? fz[leg] / fz_sum : 0.0;
      bool &st = stance_state_[leg];
      st = st ? share > params_.share_off : share > params_.share_on;
      if (!kin.converged[leg]) {
        st = false;
        n_fk_failed_++;
      }
      s.stance[leg] = st;
      s.foot_I[leg] = (R_BtoI * kin.foot[leg] + p_BinI).cast<DataType>();
      s.vrel_I[leg] = (-(double)params_.velocity_scale * R_BtoI *
                       kin.jacobian[leg] * dq.segment<3>(3 * leg))
                          .cast<DataType>();
    }
    samples_.push_back(s);
  }
  while (!samples_.empty() &&
         samples_.front().timestamp < samples_.back().timestamp - kHistory)
    samples_.pop_front();
}

bool UpdaterLegVelocity::try_update(std::shared_ptr<State> state,
                                    double timestamp) {
  last_accepted_ = 0;
  last_rejected_ = 0;
  last_residuals_.clear();

  // Gravity "up" in the body frame, from the filter's attitude (G is z-up)
  const Mat3 R_GtoI = state->imu->Rot();
  const Vec3 up_B = params_.R_BtoI.transpose() * R_GtoI * Vec3(0, 0, 1);
  process_new_samples(up_B);
  dbg_calls_++;

  // The window, on the joint clock (= IMU clock + time_offset_s)
  const double t_imu = timestamp + state->calib_dt_CAMtoIMU->value()(0);
  const double tc = t_imu + params_.time_offset_s;
  const double w = params_.half_window_s;
  if (samples_.empty() || samples_.back().timestamp < tc - params_.max_age_s ||
      samples_.back().timestamp < tc - w) {
    n_no_data_++;
    debug_print(timestamp, "STOP no recent joint sample",
                "newest sample - frame %+.3f s (needs >= %+.3f)",
                samples_.empty() ? NAN : samples_.back().timestamp - tc,
                -std::min(w, (double)params_.max_age_s));
    report(timestamp);
    return false;
  }
  const double lo = tc - w;
  const double hi = std::min(tc + w, samples_.back().timestamp);

  std::array<int, LegKinematics::kLegs> n_stance{};
  std::array<Vec3, LegKinematics::kLegs> foot, vrel;
  foot.fill(Vec3::Zero());
  vrel.fill(Vec3::Zero());
  int n = 0;
  for (const auto &s : samples_) {
    if (s.timestamp < lo || s.timestamp > hi)
      continue;
    n++;
    for (int leg = 0; leg < LegKinematics::kLegs; leg++) {
      if (!s.stance[leg])
        continue;
      n_stance[leg]++;
      foot[leg] += s.foot_I[leg];
      vrel[leg] += s.vrel_I[leg];
    }
  }
  if (n < 3) {
    n_no_data_++;
    debug_print(timestamp, "STOP too few samples in window",
                "%d samples in [%+.3f, %+.3f] s around the frame", n, lo - tc,
                hi - tc);
    report(timestamp);
    return false;
  }
  // The joint stream can run a few ms ahead of the IMU buffer; the IMU always
  // covers the frame time (the frame is processed once the IMU has passed it)
  Vec3 w_mean;
  const bool gyro_full =
      gyro_ && gyro_(lo - params_.time_offset_s, hi - params_.time_offset_s,
                     w_mean);
  if (!gyro_full &&
      (!gyro_ || !gyro_(lo - params_.time_offset_s,
                        std::min(hi, tc) - params_.time_offset_s, w_mean))) {
    n_no_gyro_++;
    debug_print(timestamp, "STOP no gyro over window",
                "IMU does not cover [%+.3f, %+.3f] s nor [%+.3f, %+.3f] s "
                "around the frame",
                lo - tc, hi - tc, lo - tc, std::min(hi, tc) - tc);
    report(timestamp);
    return false;
  }

  //=========================================================================
  // Residuals and Jacobians of the stance legs
  //=========================================================================
  const bool use_fej = state->options.do_fej;
  const Mat3 R_jac = use_fej ? state->imu->Rot_fej() : R_GtoI;
  const Vec3 v_jac = use_fej ? state->imu->vel_fej() : state->imu->vel();
  const Vec3 v = state->imu->vel();
  const Vec3 bg = state->imu->bias_g();

  std::vector<std::shared_ptr<ov_type::Type>> Hx_order;
  Hx_order.push_back(state->imu->q());
  Hx_order.push_back(state->imu->v());
  Hx_order.push_back(state->imu->bg());

  std::vector<int> legs;
  std::vector<Vec3> res;
  std::vector<Eigen::Matrix<DataType, 3, 9>> Hs;
  for (int leg = 0; leg < LegKinematics::kLegs; leg++) {
    if (n_stance[leg] < params_.min_stance_fraction * n)
      continue;
    const Vec3 f = foot[leg] / (DataType)n_stance[leg];
    const Vec3 m = vrel[leg] / (DataType)n_stance[leg] + skew_x(f) * w_mean;
    const Vec3 h = R_GtoI * v + skew_x(f) * bg;
    Eigen::Matrix<DataType, 3, 9> H;
    // JPL left error R = (I - [dθ]x) R̂: a rotated vector picks up +[R̂ v]x
    H.block<3, 3>(0, 0) = skew_x(Vec3(R_jac * v_jac));
    H.block<3, 3>(0, 3) = R_jac;
    H.block<3, 3>(0, 6) = skew_x(f);
    legs.push_back(leg);
    res.push_back(m - h);
    Hs.push_back(H);
  }
  last_residuals_ = res;
  if (legs.empty()) {
    debug_print(timestamp, "STOP no leg in stance",
                "stance samples per leg %d %d %d %d of %d (need %.0f), gyro %s",
                n_stance[0], n_stance[1], n_stance[2], n_stance[3], n,
                params_.min_stance_fraction * n,
                gyro_full ? "full" : "fallback");
    report(timestamp);
    return false;
  }

  //=========================================================================
  // Per-leg chi2 gate, then one square-root update with the accepted legs
  //=========================================================================
  MatX U_dense, U_tri;
  StateHelper::get_marginal_U_block(state, Hx_order, U_dense, U_tri);
  const DataType sigma = params_.sigma_v;
  const DataType chi2_check = options_.chi2_multipler * chi_squared_table_[3];

  std::vector<int> accepted;
  std::vector<MatX> HUTs;
  for (size_t i = 0; i < legs.size(); i++) {
    MatX HUT = MatX::Zero(3, U_dense.rows() + U_tri.rows());
    HUT.leftCols(U_dense.rows()).noalias() = Hs[i] * U_dense.transpose();
    HUT.rightCols(U_tri.rows()).noalias() =
        Hs[i] * U_tri.transpose().triangularView<Eigen::Lower>();
    Mat3 S = HUT * HUT.transpose();
    S.diagonal().array() += sigma * sigma;
    const DataType chi2 = res[i].dot(S.llt().solve(res[i]));
    if (chi2 > chi2_check) {
      PRINT_DEBUG(YELLOW "[LEG]: leg %d rejected (chi2 %.2f > %.2f, res "
                         "%.3f %.3f %.3f m/s)\n" RESET,
                  legs[i], chi2, chi2_check, res[i](0), res[i](1), res[i](2));
      last_rejected_++;
      continue;
    }
    accepted.push_back((int)i);
    HUTs.push_back(HUT);
  }
  n_legs_rejected_ += last_rejected_;
  if (accepted.empty()) {
    // Every stance leg disagrees with the state: either all feet slip at once,
    // or (far likelier) the state ran away while staying confident. After a
    // while, give the velocity back enough uncertainty for the legs to pull it.
    if (inconsistent_since_ < 0)
      inconsistent_since_ = timestamp;
    if (params_.recover_after_s > 0 &&
        timestamp - inconsistent_since_ >= params_.recover_after_s) {
      DataType worst = 0.0;
      for (const auto &r : res)
        worst = std::max(worst, r.norm());
      const DataType sigma_add =
          std::min((DataType)5.0, std::max((DataType)0.2, (DataType)2.0 * worst));
      StateHelper::inflate_velocity(state, sigma_add);
      PRINT_INFO(YELLOW "[LEG]: every stance leg failed the gate for %.1f s "
                        "(worst residual %.2f m/s) - re-acquiring with %.2f "
                        "m/s of added velocity uncertainty\n" RESET,
                 timestamp - inconsistent_since_, worst, sigma_add);
      inconsistent_since_ = -1;
      n_recoveries_++;
    }
    debug_print(timestamp, "STOP every stance leg failed the gate",
                "%zu stance legs, first residual %.3f m/s", legs.size(),
                (double)res[0].norm());
    report(timestamp);
    return false;
  }
  inconsistent_since_ = -1;

  const int rows = 3 * (int)accepted.size();
  MatX HUT_all(rows, HUTs[0].cols());
  VecX RHTr = VecX::Zero(state->get_state_size());
  for (size_t a = 0; a < accepted.size(); a++) {
    const int i = accepted[a];
    HUT_all.middleRows(3 * a, 3) = HUTs[a] / sigma; // whiten: R^(-1/2) H U^T
    const Vec3 R_inv_res = res[i] / (sigma * sigma);
    int local_id = 0;
    for (const auto &var : Hx_order) {
      RHTr.middleRows(var->id(), var->size()).noalias() +=
          Hs[i].block(0, local_id, 3, var->size()).transpose() * R_inv_res;
      local_id += var->size();
    }
  }

  // Joint samples arrive on their own schedule, so this update stands on its
  // own, like the sun update, instead of joining the per-frame batch.
  state->setup_matrix_buffer();
  state->store_update_factor(HUT_all, RHTr);
  StateHelper::update_llt(state);
  state->clear();

  last_accepted_ = (int)accepted.size();
  n_legs_accepted_ += last_accepted_;
  n_updates_++;
  debug_print(timestamp, "OK update applied",
              "%d legs accepted, %d rejected, %d samples, gyro %s",
              last_accepted_, last_rejected_, n,
              gyro_full ? "full" : "fallback");
  report(timestamp);
  return true;
}

void UpdaterLegVelocity::debug_print(double timestamp, const char *stage,
                                     const char *fmt, ...) {
  if (dbg_last_print_ >= 0 && timestamp - dbg_last_print_ < 1.0)
    return;
  char detail[256];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(detail, sizeof(detail), fmt, args);
  va_end(args);
  int fed;
  {
    std::lock_guard<std::mutex> lck(raw_mtx_);
    fed = dbg_fed_;
    dbg_fed_ = 0;
  }
  PRINT_INFO("[LEG-DBG]: %s | %s | since last: %d calls, %d msgs fed, %d "
             "processed, %d out of order, %d missing vel/effort, %zu samples "
             "held\n",
             stage, detail, dbg_calls_, fed, dbg_processed_, dbg_old_,
             dbg_bad_size_, samples_.size());
  dbg_last_print_ = timestamp;
  dbg_calls_ = dbg_processed_ = dbg_old_ = dbg_bad_size_ = 0;
}

void UpdaterLegVelocity::report(double timestamp) {
  if (stats_since_ < 0)
    stats_since_ = timestamp;
  if (timestamp - stats_since_ < kStatsPeriod)
    return;
  PRINT_INFO(CYAN "[LEG]: last %.0f s: %d updates, %d legs applied, %d "
                  "rejected by the gate, %d frames without joint data, %d "
                  "without gyro, %d leg solves failed, %d recoveries\n" RESET,
             timestamp - stats_since_, n_updates_, n_legs_accepted_,
             n_legs_rejected_, n_no_data_, n_no_gyro_, n_fk_failed_,
             n_recoveries_);
  stats_since_ = timestamp;
  n_updates_ = n_no_data_ = n_no_gyro_ = n_legs_accepted_ = n_legs_rejected_ =
      n_fk_failed_ = n_recoveries_ = 0;
}
