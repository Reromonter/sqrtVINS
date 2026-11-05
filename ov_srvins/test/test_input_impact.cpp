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

// The per-frame input account (InputImpact) against a synthetic State: frame
// conventions of the yaw / tilt split, the correction lengths, the telescoping
// of the information steps, and the sign of propagation, a sun update and a
// recovery inflation.

#include <cmath>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "initializer/InertialInitializerOptions.h"
#include "state/InputImpact.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "state/StateOptions.h"
#include "update/UpdaterOptions.h"
#include "update/UpdaterSunSensor.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_srvins;

namespace {

constexpr double kDeg = M_PI / 180.0;
const DataType kAz = 120.0 * kDeg;
const DataType kEl = 35.0 * kDeg;

/// IMU 1-sigmas: theta (rad), p (m), v (m/s), bg, ba
const DataType kSigTheta = 0.01, kSigP = 0.1, kSigV = 0.05, kSigBg = 0.001,
               kSigBa = 0.01;

std::shared_ptr<State> make_state(bool calib_sun = false) {
  ov_core::Printer::setPrintLevel("WARNING");
  StateOptions so;
  so.do_fej = false;
  so.num_cameras = 1;
  so.do_calib_sun_align = calib_sun;
  InertialInitializerOptions io;
  io.init_prior_sun_align = 0.5;
  auto state = std::make_shared<State>(so, io);
  state->is_initialized = true;
  state->timestamp = 0.0;
  VecX diag(15);
  diag << kSigTheta, kSigTheta, kSigTheta, kSigP, kSigP, kSigP, kSigV, kSigV,
      kSigV, kSigBg, kSigBg, kSigBg, kSigBa, kSigBa, kSigBa;
  StateHelper::set_initial_imu_square_root_covariance(state, diag);
  return state;
}

void set_attitude(const std::shared_ptr<State> &state, const Mat3 &R_GtoI) {
  VecX x = state->imu->value();
  x.head(4) = ov_core::rot_2_quat(R_GtoI);
  state->imu->set_value(x);
  state->imu->set_fej(x);
}

Mat3 rot_axis(const Vec3 &axis, DataType angle) {
  return ov_core::exp_so3(Vec3(axis.normalized() * angle));
}

Vec3 ephemeris() {
  return Vec3(std::cos(kEl) * std::sin(kAz), std::cos(kEl) * std::cos(kAz),
              std::sin(kEl));
}

} // namespace

TEST(InputImpact, YawIsTheImuAxisAlongGravity) {
  // Pitched 90 deg: the IMU x axis lies along G z. Shrinking theta_x alone
  // must show as yaw information, not tilt.
  auto state = make_state();
  const Mat3 R_GtoI = rot_axis(Vec3(0, 1, 0), M_PI / 2).transpose();
  set_attitude(state, R_GtoI);
  ASSERT_NEAR(std::abs((R_GtoI * Vec3(0, 0, 1))(0)), 1.0, 1e-5);

  InputImpact::Snapshot s = InputImpact::snapshot(state);
  const auto base = InputImpact::information(s);
  EXPECT_NEAR(base(InputImpact::YAW), 1.0 / (kSigTheta * kSigTheta), 1.0);
  EXPECT_NEAR(base(InputImpact::TILT), 2.0 / (kSigTheta * kSigTheta), 1.0);
  EXPECT_NEAR(base(InputImpact::VEL), 3.0 / (kSigV * kSigV), 0.1);

  s.P(0, 0) *= 0.01; // sigma_theta_x / 10
  const auto info = InputImpact::information(s);
  EXPECT_NEAR(info(InputImpact::YAW) / base(InputImpact::YAW), 100.0, 0.5);
  EXPECT_NEAR(info(InputImpact::TILT), base(InputImpact::TILT), 1.0);
}

TEST(InputImpact, CorrectionIsReportedInGAndInSigmas) {
  auto state = make_state();
  const Mat3 R0 = rot_axis(Vec3(1, 2, 3), 0.7).transpose();
  set_attitude(state, R0);
  InputImpact impact;
  impact.begin(state, 1.0);

  // Turn the body +2 deg about G z (R_ItoG' = Rz R_ItoG) and move 0.3 m
  const DataType psi = 2.0 * kDeg;
  VecX x = state->imu->value();
  x.head(4) = ov_core::rot_2_quat(Mat3(R0 * ov_core::rot_z<DataType>(-psi)));
  x.segment(4, 3) += Vec3(0.3, 0.0, 0.0);
  state->imu->set_value(x);
  impact.mark("test", state);

  ASSERT_EQ(impact.steps().size(), 1u);
  const auto &st = impact.steps().front();
  EXPECT_NEAR(st.d_yaw_deg, 2.0, 1e-3);
  EXPECT_NEAR(st.d_tilt_deg, 0.0, 1e-3);
  EXPECT_NEAR(st.d_pos, 0.3, 1e-5);
  EXPECT_NEAR(st.d_vel, 0.0, 1e-9);
  // theta (isotropic sigma, so any axis) and p are independent in P
  const double expect = std::hypot(psi / kSigTheta, 0.3 / kSigP);
  EXPECT_NEAR(st.mahalanobis, expect, 1e-3 * expect);
  // nothing touched P
  EXPECT_NEAR(st.d_info.cwiseAbs().maxCoeff(), 0.0, 1e-2);
}

TEST(InputImpact, PropagationLosesInformationAndStepsTelescope) {
  auto state = make_state();
  InputImpact impact;
  impact.begin(state, 0.0);
  const auto start = impact.posterior();

  Eigen::Matrix<DataType, 15, 15> Phi =
      Eigen::Matrix<DataType, 15, 15>::Identity();
  Phi.block<3, 3>(3, 6) = 0.1 * Mat3::Identity(); // p += v dt
  Eigen::Matrix<DataType, 15, 15> Q_sqrt =
      0.01 * Eigen::Matrix<DataType, 15, 15>::Identity();
  StateHelper::propagate(state, Phi, Q_sqrt);
  impact.mark("imu", state, true, false);
  const auto after_prop = impact.posterior();

  StateHelper::inflate_velocity(state, 0.2);
  impact.mark("leg", state, false);

  ASSERT_EQ(impact.steps().size(), 2u);
  const auto &prop = impact.steps().at(0);
  EXPECT_EQ(prop.mahalanobis, 0.0) << "propagation records no correction";
  for (int g = 0; g < InputImpact::NUM_GROUPS; g++)
    EXPECT_LT(prop.d_info(g), 0.0) << "group " << g;

  // the inflation costs velocity information and nothing else
  const auto &leg = impact.steps().at(1);
  EXPECT_LT(leg.d_info(InputImpact::VEL), -0.5 * after_prop(InputImpact::VEL));
  EXPECT_NEAR(leg.d_info(InputImpact::YAW), 0.0,
              1e-3 * after_prop(InputImpact::YAW));

  // posterior = start + sum of the steps
  InputImpact::GroupVec sum = start;
  for (const auto &st : impact.steps())
    sum += st.d_info;
  for (int g = 0; g < InputImpact::NUM_GROUPS; g++)
    EXPECT_NEAR(sum(g), impact.posterior()(g), 1e-6 * start(g) + 1e-9);
}

TEST(InputImpact, SunUpdateBuysYawAndTheLineParses) {
  auto state = make_state(true);
  UpdaterOptions opts(1.0, 1.0);
  const DataType sigma = 0.0016;
  UpdaterSunSensor sun(opts, sigma, sigma, Mat3::Identity(), 10.0 * kDeg,
                       15.0 * kDeg, true);
  // Sensor along the IMU z axis, IMU level, alignment yaw 0.3 rad
  auto reading = [&](double t) {
    const Vec3 s = ov_core::rot_z<DataType>(0.3) * ephemeris();
    ov_core::SunSensorData m;
    m.timestamp = t;
    m.alpha = std::atan2(s(0), s(2));
    m.beta = std::atan2(s(1), s(2));
    return m;
  };
  // the first reading only initializes the alignment yaw
  sun.feed_sun(reading(0.0), 0.0);
  sun.try_update(state, 0.0, kAz, kEl);

  InputImpact impact;
  impact.begin(state, 0.2);
  sun.feed_sun(reading(0.2), 0.2);
  state->timestamp = 0.2;
  const bool ok = sun.try_update(state, 0.2, kAz, kEl);
  impact.mark("sun", state, ok);
  ASSERT_TRUE(ok);

  const auto &st = impact.steps().front();
  const auto &post = impact.posterior();
  // yaw is the ENU heading here (calib_sun_align): the 0.5 rad alignment
  // prior leaves it unknown until the sun pins it, so nearly all of the
  // posterior yaw information is the sun's
  EXPECT_GT(st.d_info(InputImpact::YAW), 0.9 * post(InputImpact::YAW));
  EXPECT_GE(st.d_info(InputImpact::TILT), 0.0);
  EXPECT_NEAR(st.d_info(InputImpact::VEL), 0.0,
              1e-3 * post(InputImpact::VEL));

  const std::string line = impact.line();
  EXPECT_EQ(line.rfind("[IMPACT]: t=0.2000 post=", 0), 0u) << line;
  EXPECT_NE(line.find(" | sun ok=1 di="), std::string::npos) << line;
  EXPECT_NE(line.find(" d="), std::string::npos) << line;
  EXPECT_NE(line.find(" c="), std::string::npos) << line;
}
