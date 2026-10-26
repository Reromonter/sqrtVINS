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

// Closed-loop checks of the sun update's time alignment, robust gate, lock-out
// recovery and heading initialization, against a synthetic State (no camera).
// The math of the measurement model itself is covered by the hand-built
// harness in lunarleaper-state-estimation/tests/cpp/test_sun_sensor_updater.cpp.

#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "initializer/InertialInitializerOptions.h"
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

Vec3 ephemeris() {
  return Vec3(std::cos(kEl) * std::sin(kAz), std::cos(kEl) * std::cos(kAz),
              std::sin(kEl));
}

Vec2 angles(const Vec3 &s) {
  return Vec2(std::atan2(s(0), s(2)), std::atan2(s(1), s(2)));
}

/// A filter state plus a sun updater, sensor boresight along the IMU z axis
struct Rig {
  std::shared_ptr<State> state;
  std::shared_ptr<UpdaterSunSensor> sun;
  /// psi_align the readings are generated with (G-to-ENU yaw)
  DataType psi_true = 0.3;

  explicit Rig(DataType sigma = 0.0016, bool calib_align = true) {
    ov_core::Printer::setPrintLevel("WARNING");
    StateOptions so;
    so.do_fej = false;
    so.num_cameras = 1;
    so.do_calib_sun_align = calib_align;
    InertialInitializerOptions io;
    io.init_prior_sun_align = 0.5;
    state = std::make_shared<State>(so, io);
    state->is_initialized = true;
    state->timestamp = 0.0;
    UpdaterOptions opts(1.0, 1.0);
    sun = std::make_shared<UpdaterSunSensor>(opts, sigma, sigma,
                                             Mat3::Identity(), 10.0 * kDeg,
                                             15.0 * kDeg, true);
  }

  void set_attitude(const Mat3 &R_GtoI) {
    VecX x = state->imu->value();
    x.head(4) = ov_core::rot_2_quat(R_GtoI);
    state->imu->set_value(x);
    state->imu->set_fej(x);
  }

  /// What the sensor reads with the IMU at R_GtoI
  ov_core::SunSensorData reading(double t, const Mat3 &R_GtoI,
                                 DataType extra_yaw = 0.0) const {
    const Vec2 a =
        angles(R_GtoI * ov_core::rot_z<DataType>(psi_true + extra_yaw) *
               ephemeris());
    ov_core::SunSensorData m;
    m.timestamp = t;
    m.alpha = a(0);
    m.beta = a(1);
    return m;
  }

  bool update(double t_state) {
    state->timestamp = t_state;
    return sun->try_update(state, t_state, kAz, kEl);
  }

  /// One-shot init plus `n` consistent readings at 5 Hz, IMU at identity:
  /// afterwards the heading is known to well under the sensor noise.
  void converge(int n = 25) {
    for (int k = 0; k <= n; k++) {
      const double t = 0.2 * k;
      sun->feed_sun(reading(t, Mat3::Identity()), t);
      update(t);
    }
  }
};

/// Body rotating at `w` rad/s about the IMU z axis: R_GtoI(t) = exp(-w t) R0
Mat3 spun(DataType w, double t) {
  return ov_core::exp_so3(Vec3(Vec3(0.0, 0.0, -w * (DataType)t)));
}

} // namespace

TEST(SunSensorUpdate, GyroAlignmentRemovesTheReadingStateSkew) {
  const DataType w = 30.0 * kDeg; // a brisk turn
  for (bool align : {false, true}) {
    Rig rig;
    rig.converge();
    rig.sun->set_time_align(align, [&](double, double, Vec3 &w_mean) {
      w_mean = Vec3(0.0, 0.0, w);
      return true;
    });
    // camera clock 10.0; the IMU clock leads it by 6 ms. Readings on the 40 Hz
    // grid either side; the state's attitude is the one at IMU time 10.006.
    VecX dt(1);
    dt(0) = 0.006;
    rig.state->calib_dt_CAMtoIMU->set_value(dt);
    const double t_imu = 10.006;
    rig.set_attitude(spun(w, t_imu));
    for (double t : {9.9925, 10.0175})
      rig.sun->feed_sun(rig.reading(t, spun(w, t)), 9.9);
    rig.update(10.0);
    const DataType res_deg = rig.sun->last_residual().norm() / kDeg;
    if (align)
      EXPECT_LT(res_deg, 0.01) << "aligned residual should vanish";
    else
      EXPECT_GT(res_deg, 0.1) << "the unaligned reading is ~10 ms off";
  }
}

TEST(SunSensorUpdate, RobustGateDownWeightsUpToTheHardLimit) {
  // 1 deg of heading disagreement against a state converged to ~0.02 deg
  struct Case {
    bool robust;
    DataType max_res_deg;
    bool want_used;
  };
  for (const Case &c : {Case{false, 5.0, false}, Case{true, 5.0, true},
                        Case{true, 0.3, false}}) {
    Rig rig;
    rig.converge();
    rig.sun->set_robust(c.robust, c.max_res_deg * kDeg);
    rig.sun->feed_sun(rig.reading(6.0, Mat3::Identity(), 1.0 * kDeg), 5.9);
    const bool used = rig.update(6.0);
    const DataType before = rig.sun->last_residual().norm();
    EXPECT_GT(before, 1.0 * kDeg); // the reading really is ~1.4 deg off
    EXPECT_EQ(used, c.want_used) << "robust " << c.robust << " max "
                                 << c.max_res_deg;
    if (used) {
      // pulled towards the reading, but by a bounded step, not all the way
      rig.sun->feed_sun(rig.reading(6.2, Mat3::Identity(), 1.0 * kDeg), 6.1);
      rig.update(6.2);
      const DataType after = rig.sun->last_residual().norm();
      EXPECT_LT(after, before);
      EXPECT_GT(after, 0.2 * before);
    }
  }
}

TEST(SunSensorUpdate, RecoveryEscapesTheLockOut) {
  // The heading jumps by 3 deg (a drift the gate never saw happen). Without
  // recovery the converged filter rejects it forever; with it, the filter
  // re-acquires and ends up agreeing with the sun again.
  for (DataType recover_after : {0.0, 1.0}) {
    Rig rig;
    rig.converge();
    rig.sun->set_recovery(recover_after);
    int used = 0;
    for (int k = 1; k <= 40; k++) {
      const double t = 5.0 + 0.2 * k;
      rig.sun->feed_sun(rig.reading(t, Mat3::Identity(), 3.0 * kDeg), t - 0.1);
      used += rig.update(t);
    }
    const DataType final_deg = rig.sun->last_residual().norm() / kDeg;
    if (recover_after > 0) {
      EXPECT_GT(used, 20);
      EXPECT_LT(final_deg, 0.05);
    } else {
      EXPECT_EQ(used, 0);
      EXPECT_GT(final_deg, 1.0);
    }
  }
}

TEST(SunSensorUpdate, InflateGlobalYawAddsExactlyTheYawDirection) {
  Rig rig;
  const Mat3 R_GtoI = ov_core::exp_so3(Vec3(0.3, -0.2, 1.1));
  VecX x = rig.state->imu->value();
  x.head(4) = ov_core::rot_2_quat(R_GtoI);
  x.segment(4, 3) = Vec3(2.0, -1.0, 0.5);  // p_IinG
  x.segment(7, 3) = Vec3(0.3, 0.4, -0.1);  // v_IinG
  rig.state->imu->set_value(x);
  rig.state->imu->set_fej(x);
  std::vector<std::shared_ptr<ov_type::Type>> order{
      rig.state->imu->q(), rig.state->imu->p(), rig.state->imu->v()};

  const MatX P0 = StateHelper::get_marginal_covariance(rig.state, order);
  const DataType sigma = 0.1;
  StateHelper::inflate_global_yaw(rig.state, sigma);
  const MatX P1 = StateHelper::get_marginal_covariance(rig.state, order);

  const Vec3 e_z(0.0, 0.0, 1.0);
  VecX n(9);
  n << R_GtoI * e_z, e_z.cross(Vec3(2.0, -1.0, 0.5)),
      e_z.cross(Vec3(0.3, 0.4, -0.1));
  const MatX want = sigma * sigma * n * n.transpose();
  EXPECT_LT((P1 - P0 - want).cwiseAbs().maxCoeff(), 1e-4);
}

namespace {

/// Angle (deg) between two rotations
DataType angle_deg(const Mat3 &A, const Mat3 &B) {
  return ov_core::log_so3(Mat3(A * B.transpose())).norm() / kDeg;
}

} // namespace

TEST(SunSensorUpdate, RotateGlobalYawIsAChangeOfGauge) {
  Rig rig;
  const Mat3 R_clone = ov_core::exp_so3(Vec3(0.3, -0.2, 1.1));
  const Vec3 p_clone(2.0, -1.0, 0.5);
  VecX x = rig.state->imu->value();
  x.head(4) = ov_core::rot_2_quat(R_clone);
  x.segment(4, 3) = p_clone;
  rig.state->imu->set_value(x);
  rig.state->imu->set_fej(x);
  auto clone = std::dynamic_pointer_cast<ov_type::PoseJPL>(
      StateHelper::clone(rig.state, rig.state->imu->pose()->clone()));
  ASSERT_NE(clone, nullptr);
  rig.state->clones_IMU[0.0] = clone;

  // A fresh clone is an exact copy of the IMU pose (rank-deficient U) until
  // propagation adds noise. Propagate once with a mixing Phi, so the blocks
  // are distinct and cross-correlated.
  std::srand(1);
  const Eigen::Matrix<DataType, 15, 15> Phi =
      Eigen::Matrix<DataType, 15, 15>::Identity() +
      0.2 * Eigen::Matrix<DataType, 15, 15>::Random();
  const Eigen::Matrix<DataType, 15, 15> Q_sqrt =
      0.01 * Eigen::Matrix<DataType, 15, 15>::Identity();
  StateHelper::propagate(rig.state, Phi, Q_sqrt);

  // Move the IMU on, so the clone and the current pose differ
  const Mat3 R_imu = ov_core::exp_so3(Vec3(-0.1, 0.25, -0.6));
  const Vec3 p_imu(3.0, 1.0, 0.2), v_imu(0.3, 0.4, -0.1), bg(0.01, 0.0, -0.02);
  x.head(4) = ov_core::rot_2_quat(R_imu);
  x.segment(4, 3) = p_imu;
  x.segment(7, 3) = v_imu;
  x.segment(10, 3) = bg;
  rig.state->imu->set_value(x);
  rig.state->imu->set_fej(x);

  std::vector<std::shared_ptr<ov_type::Type>> order{
      rig.state->imu->q(), rig.state->imu->p(), rig.state->imu->v(),
      clone->q(), clone->p()};
  const MatX P0 = StateHelper::get_marginal_covariance(rig.state, order);

  const DataType dpsi = 0.7;
  StateHelper::rotate_global_yaw(rig.state, dpsi);
  const Mat3 C = ov_core::rot_z<DataType>(dpsi);

  EXPECT_LT(angle_deg(rig.state->imu->Rot(), R_imu * C.transpose()), 0.01);
  EXPECT_LT(angle_deg(rig.state->imu->Rot_fej(), R_imu * C.transpose()), 0.01);
  EXPECT_LT((rig.state->imu->pos() - C * p_imu).norm(), 1e-5);
  EXPECT_LT((rig.state->imu->vel() - C * v_imu).norm(), 1e-5);
  EXPECT_LT((rig.state->imu->bias_g() - bg).norm(), 1e-7);
  EXPECT_LT(angle_deg(clone->Rot(), R_clone * C.transpose()), 0.01);
  EXPECT_LT(angle_deg(clone->Rot_fej(), R_clone * C.transpose()), 0.01);
  EXPECT_LT((clone->pos() - C * p_clone).norm(), 1e-5);

  // P' = T P T^T: orientation errors stay put, positions and velocity turn
  MatX T = MatX::Identity(15, 15);
  for (int block : {1, 2, 4})
    T.block(3 * block, 3 * block, 3, 3) = C;
  const MatX P1 = StateHelper::get_marginal_covariance(rig.state, order);
  EXPECT_LT((P1 - T * P0 * T.transpose()).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(SunSensorUpdate, InitHeadingStartsInEnuWithoutAlignmentCalibration) {
  // calib_sun_align off and G 57 deg away from ENU. Without init_heading the
  // filter's tight heading prior rejects a 57 deg residual, so the sun is
  // locked out from the start. With it, the first reading turns G onto ENU
  // and every later reading is used.
  const Mat3 R_GtoI = ov_core::rot_z<DataType>(0.4);
  for (bool init_heading : {false, true}) {
    Rig rig(0.0016, false);
    rig.psi_true = 1.0;
    rig.sun->set_init_heading(init_heading);
    rig.set_attitude(R_GtoI);
    int used = 0;
    for (int k = 0; k <= 25; k++) {
      const double t = 0.2 * k;
      rig.sun->feed_sun(rig.reading(t, R_GtoI), t);
      used += rig.update(t);
    }
    if (init_heading) {
      EXPECT_EQ(used, 25) << "every reading after the first is used";
      EXPECT_LT(rig.sun->last_residual().norm() / kDeg, 0.05);
      // The estimate is now R_ENUtoI = R_GtoI * rot_z(psi_true)
      EXPECT_LT(angle_deg(rig.state->imu->Rot(),
                          R_GtoI * ov_core::rot_z<DataType>(rig.psi_true)),
                0.05);
      EXPECT_EQ(rig.state->calib_sun_align_yaw->value()(0), 0.0);
    } else {
      EXPECT_EQ(used, 0);
      EXPECT_GT(rig.sun->last_residual().norm() / kDeg, 10.0);
    }
  }
}

TEST(SunSensorUpdate, InitHeadingWithCalibrationZeroesTheAlignmentYaw) {
  // Same start with the alignment yaw calibrated: init_heading puts the
  // heading into the IMU and leaves psi_align at zero, instead of the other
  // way round.
  const Mat3 R_GtoI = ov_core::rot_z<DataType>(0.4);
  for (bool init_heading : {false, true}) {
    Rig rig;
    rig.psi_true = 1.0;
    rig.sun->set_init_heading(init_heading);
    rig.set_attitude(R_GtoI);
    rig.sun->feed_sun(rig.reading(0.0, R_GtoI), 0.0);
    rig.update(0.0);
    const DataType psi = rig.state->calib_sun_align_yaw->value()(0);
    if (init_heading) {
      EXPECT_NEAR(psi, 0.0, 1e-6);
      EXPECT_LT(angle_deg(rig.state->imu->Rot(),
                          R_GtoI * ov_core::rot_z<DataType>(rig.psi_true)),
                0.01);
    } else {
      EXPECT_NEAR(psi, rig.psi_true, 1e-4);
      EXPECT_LT(angle_deg(rig.state->imu->Rot(), R_GtoI), 0.01);
    }
  }
}
