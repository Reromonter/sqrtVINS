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

#include "InputImpact.h"

#include <cmath>
#include <cstdio>

using namespace ov_srvins;

namespace {

constexpr double kRadToDeg = 180.0 / M_PI;

/// tr(B^-1) of a symmetric block; 0 if it is not positive definite
template <typename Block> double trace_inverse(const Block &B) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(B);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive() ||
      ldlt.vectorD().minCoeff() <= 0.0)
    return 0.0;
  return ldlt.solve(Eigen::MatrixXd::Identity(B.rows(), B.cols())).trace();
}

/// Rotation vector of R (log map)
Eigen::Vector3d log_so3(const Eigen::Matrix3d &R) {
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}

} // namespace

InputImpact::Snapshot
InputImpact::snapshot(const std::shared_ptr<State> &state) {
  Snapshot s;
  s.R_GtoI = state->imu->Rot().cast<double>();
  s.p = state->imu->pos().cast<double>();
  s.v = state->imu->vel().cast<double>();
  s.bg = state->imu->bias_g().cast<double>();
  s.ba = state->imu->bias_a().cast<double>();
  // P = U^T U over the IMU (and alignment) columns
  s.has_align = state->options.do_calib_sun_align;
  Eigen::MatrixXd U_cols = Eigen::MatrixXd::Zero(state->U_.rows(), 16);
  U_cols.leftCols(15) =
      state->U_.middleCols(state->imu->id(), 15).cast<double>();
  if (s.has_align) {
    s.psi_align = (double)state->calib_sun_align_yaw->value()(0);
    U_cols.col(15) =
        state->U_.col(state->calib_sun_align_yaw->id()).cast<double>();
  }
  s.P = U_cols.transpose() * U_cols;
  return s;
}

InputImpact::GroupVec InputImpact::information(const Snapshot &s) {
  // theta_G = R_GtoI^T theta_I (JPL: R_true = (I - [theta_I]x) R_GtoI)
  Mat16d T = Mat16d::Identity();
  T.topLeftCorner<3, 3>() = s.R_GtoI.transpose();
  const Mat16d P = T * s.P * T.transpose();
  GroupVec info;
  // heading in ENU = theta_Gz - psi (the sun sits at rot_z(psi) s_ENU in G)
  const double var_yaw =
      s.has_align ? P(2, 2) - 2.0 * P(2, 15) + P(15, 15) : P(2, 2);
  info(YAW) = trace_inverse(Eigen::Matrix<double, 1, 1>(var_yaw));
  info(TILT) = trace_inverse(P.block<2, 2>(0, 0));
  info(POS_XY) = trace_inverse(P.block<2, 2>(3, 3));
  info(POS_Z) = trace_inverse(P.block<1, 1>(5, 5));
  info(VEL) = trace_inverse(P.block<3, 3>(6, 6));
  info(BG) = trace_inverse(P.block<3, 3>(9, 9));
  info(BA) = trace_inverse(P.block<3, 3>(12, 12));
  return info;
}

void InputImpact::begin(const std::shared_ptr<State> &state,
                        double timestamp) {
  timestamp_ = timestamp;
  steps_.clear();
  last_ = snapshot(state);
  info_last_ = information(last_);
}

void InputImpact::mark(const std::string &source,
                       const std::shared_ptr<State> &state, bool accepted,
                       bool is_update) {
  const Snapshot now = snapshot(state);
  const GroupVec info_now = information(now);

  Step step;
  step.source = source;
  step.accepted = accepted;
  step.d_info = info_now - info_last_;
  if (is_update) {
    // Error state between the snapshots, in the filter's ordering
    Vec16d dx;
    const Eigen::Vector3d dtheta_I =
        -log_so3(now.R_GtoI * last_.R_GtoI.transpose());
    dx.segment<3>(0) = dtheta_I;
    dx.segment<3>(3) = now.p - last_.p;
    dx.segment<3>(6) = now.v - last_.v;
    dx.segment<3>(9) = now.bg - last_.bg;
    dx.segment<3>(12) = now.ba - last_.ba;
    dx(15) = now.psi_align - last_.psi_align;
    const int n = last_.has_align ? 16 : 15;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(last_.P.topLeftCorner(n, n));
    if (ldlt.info() == Eigen::Success && dx.norm() > 0.0)
      step.mahalanobis = std::sqrt(
          std::max(0.0, dx.head(n).dot(ldlt.solve(dx.head(n)))));
    const Eigen::Vector3d dtheta_G = last_.R_GtoI.transpose() * dtheta_I;
    step.d_yaw_deg = (dtheta_G(2) - dx(15)) * kRadToDeg;
    step.d_tilt_deg = dtheta_G.head<2>().norm() * kRadToDeg;
    step.d_pos = dx.segment<3>(3).norm();
    step.d_vel = dx.segment<3>(6).norm();
  }
  steps_.push_back(step);
  last_ = now;
  info_last_ = info_now;
}

std::string InputImpact::line() const {
  char buf[256];
  std::string out;
  std::snprintf(buf, sizeof(buf), "[IMPACT]: t=%.4f post=", timestamp_);
  out += buf;
  for (int g = 0; g < NUM_GROUPS; g++) {
    std::snprintf(buf, sizeof(buf), g == 0 ? "%.4g" : ",%.4g", info_last_(g));
    out += buf;
  }
  for (const auto &s : steps_) {
    std::snprintf(buf, sizeof(buf), " | %s ok=%d di=", s.source.c_str(),
                  (int)s.accepted);
    out += buf;
    for (int g = 0; g < NUM_GROUPS; g++) {
      std::snprintf(buf, sizeof(buf), g == 0 ? "%.4g" : ",%.4g", s.d_info(g));
      out += buf;
    }
    std::snprintf(buf, sizeof(buf), " d=%.3g c=%.4g,%.4g,%.4g,%.4g",
                  s.mahalanobis, s.d_yaw_deg, s.d_tilt_deg, s.d_pos, s.d_vel);
    out += buf;
  }
  return out;
}
