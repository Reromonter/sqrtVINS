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

// Leg kinematics against the Python solver it was ported from, and closed-loop
// checks of the leg velocity update on a synthetic State (no clones, no
// camera). Test data: test/data/leg_mechanism_bwl.yaml and the Python results
// in leg_kinematics_reference_bwl.yaml, both written by
// lunarleaper-state-estimation/legged_odometry/tools/export_mechanism.py.

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core/persistence.hpp>

#include "initializer/InertialInitializerOptions.h"
#include "legs/LegKinematics.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "state/StateOptions.h"
#include "update/UpdaterLegVelocity.h"
#include "update/UpdaterOptions.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_srvins;

namespace {

const std::string kData = LEG_TEST_DATA_DIR;
const std::string kMechanism = kData + "/leg_mechanism_bwl.yaml";
typedef Eigen::Matrix<double, LegKinematics::kMotors, 1> Motors;

/// The policies' default stance, (T, L, R) per leg
Motors stance() {
  Motors q;
  for (int leg = 0; leg < 4; leg++)
    q.segment<3>(3 * leg) << 0.0, -0.785, 0.785;
  return q;
}

} // namespace

TEST(LegKinematics, MatchesThePythonSolver) {
  cv::FileStorage fs(kData + "/leg_kinematics_reference_bwl.yaml",
                     cv::FileStorage::READ);
  ASSERT_TRUE(fs.isOpened());
  int n = 0;
  fs["num_samples"] >> n;
  std::vector<double> q, foot, jac;
  fs["q"] >> q;
  fs["foot"] >> foot;
  fs["jacobian"] >> jac;
  ASSERT_EQ((int)q.size(), 12 * n);

  LegKinematics kin(kMechanism);
  for (int k = 0; k < n; k++) {
    Motors qk = Eigen::Map<const Motors>(q.data() + 12 * k);
    LegKinematics::Result r;
    ASSERT_TRUE(kin.solve(qk, r)) << "sample " << k;
    for (int leg = 0; leg < 4; leg++) {
      const Eigen::Vector3d p(foot.data() + 12 * k + 3 * leg);
      // the Python export is row-major (x, y, z rows of T, L, R columns)
      const Eigen::Matrix3d J = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          jac.data() + 36 * k + 9 * leg);
      // The USD authors float quaternions (7 digits); the Python model uses
      // them as written, this one normalizes them: sub-micron differences.
      EXPECT_LT((r.foot[leg] - p).norm(), 2e-6) << "sample " << k << " leg " << leg;
      EXPECT_LT((r.jacobian[leg] - J).norm(), 2e-5) << "sample " << k << " leg " << leg;
    }
  }
}

TEST(LegKinematics, JacobianMatchesFiniteDifferences) {
  LegKinematics kin(kMechanism);
  Motors q0 = stance();
  q0(1) += 0.1;
  q0(9) -= 0.2;
  LegKinematics::Result r0;
  ASSERT_TRUE(kin.solve(q0, r0));
  const double h = 1e-6;
  for (int i = 0; i < 12; i++) {
    Motors qp = q0, qm = q0;
    qp(i) += h;
    qm(i) -= h;
    LegKinematics::Result rp, rm;
    kin.solve(qp, rp);
    kin.solve(qm, rm);
    const int leg = i / 3;
    const Eigen::Vector3d d = (rp.foot[leg] - rm.foot[leg]) / (2 * h);
    EXPECT_LT((d - r0.jacobian[leg].col(i % 3)).norm(), 1e-6) << "motor " << i;
    for (int other = 0; other < 4; other++)
      if (other != leg)
        EXPECT_LT((rp.foot[other] - rm.foot[other]).norm(), 1e-12);
  }
}

TEST(LegKinematics, UnreachableLegHoldsThenRecovers) {
  LegKinematics kin(kMechanism);
  LegKinematics::Result good, bad, again;
  ASSERT_TRUE(kin.solve(stance(), good));
  Motors q = stance();
  q.segment<3>(0) << 0.0, 1.5, 1.5; // both arms swung the same way: cannot close
  EXPECT_FALSE(kin.solve(q, bad));
  EXPECT_FALSE(bad.converged[0]);
  EXPECT_TRUE(bad.converged[1] && bad.converged[2] && bad.converged[3]);
  ASSERT_TRUE(kin.solve(stance(), again));
  for (int leg = 0; leg < 4; leg++)
    EXPECT_LT((again.foot[leg] - good.foot[leg]).norm(), 1e-9);
}

namespace {

/// A filter state and a leg updater, body = IMU frame, gyro at rest
struct Rig {
  std::shared_ptr<State> state;
  std::shared_ptr<LegKinematics> kin;
  std::shared_ptr<UpdaterLegVelocity> leg;
  Vec3 w_true = Vec3::Zero();

  explicit Rig(DataType sigma = 0.05, double recover_after_s = 1.0) {
    ov_core::Printer::setPrintLevel("WARNING");
    StateOptions so;
    so.do_fej = false;
    so.num_cameras = 1;
    InertialInitializerOptions io;
    state = std::make_shared<State>(so, io);
    state->is_initialized = true;
    state->timestamp = 0.0;
    // the initializer's prior (sqrt covariance): q, p, v, bg, ba
    VecX prior = VecX::Ones(15);
    prior.segment<3>(0) *= 0.017;
    prior.segment<3>(3) *= 0.05;
    prior.segment<3>(6) *= 0.5;
    prior.segment<3>(9) *= 0.02;
    prior.segment<3>(12) *= 0.02;
    StateHelper::set_initial_imu_square_root_covariance(state, prior);
    kin = std::make_shared<LegKinematics>(kMechanism);
    UpdaterOptions opts(1.0, 1.0);
    UpdaterLegVelocity::Params p;
    p.sigma_v = sigma;
    p.recover_after_s = recover_after_s;
    leg = std::make_shared<UpdaterLegVelocity>(
        opts, p, kin, [this](double, double, Vec3 &w) {
          w = w_true;
          return true;
        });
  }

  /// One joint sample of the robot moving at v_B with the diagonal pair
  /// (LF + RH or RF + LH) loaded; `slip` adds a velocity to one leg's foot
  ov_core::JointStateData sample(double t, const Eigen::Vector3d &v_B,
                                 bool lf_rh, int slip_leg = -1,
                                 const Eigen::Vector3d &slip = Eigen::Vector3d::Zero()) {
    LegKinematics probe(kMechanism);
    LegKinematics::Result r;
    probe.solve(stance(), r);
    ov_core::JointStateData m;
    m.timestamp = t;
    m.names = kin->motor_names();
    for (int leg = 0; leg < 4; leg++) {
      const bool loaded = ((leg == 0 || leg == 3) == lf_rh);
      // a pinned foot moves at -v_B relative to the body (no rotation here)
      Eigen::Vector3d foot_vel = -v_B;
      if (leg == slip_leg)
        foot_vel += slip;
      const Eigen::Vector3d dq = r.jacobian[leg].fullPivLu().solve(foot_vel);
      // torques that hold a vertical ground force of 100 N, or nothing
      const Eigen::Vector3d F(0.0, 0.0, loaded ? 100.0 : 0.0);
      const Eigen::Vector3d tau = -r.jacobian[leg].transpose() * F;
      for (int k = 0; k < 3; k++) {
        m.position.push_back(stance()(3 * leg + k));
        m.velocity.push_back(dq(k));
        m.effort.push_back(tau(k));
      }
    }
    return m;
  }

  /// Feed 200 Hz joint samples over [t - 0.15, t + 0.15] and update at t
  bool step(double t, const Eigen::Vector3d &v_B, int slip_leg = -1,
            const Eigen::Vector3d &slip = Eigen::Vector3d::Zero()) {
    for (int k = -30; k <= 30; k++)
      leg->feed_joint_state(sample(t + 0.005 * k, v_B, true, slip_leg, slip), -1);
    state->timestamp = t;
    return leg->try_update(state, t);
  }
};

} // namespace

TEST(LegVelocityUpdate, StanceLegsPullTheVelocityToTheTruth) {
  Rig rig;
  const Eigen::Vector3d v_true(0.3, -0.1, 0.02);
  ASSERT_TRUE(rig.step(1.0, v_true));
  EXPECT_EQ(rig.leg->last_accepted(), 2) << "only the loaded diagonal is in stance";
  for (int k = 2; k <= 20; k++)
    rig.step(0.5 * k, v_true);
  const Vec3 v = rig.state->imu->vel();
  EXPECT_LT((v - v_true.cast<DataType>()).norm(), 0.01)
      << "v = " << v.transpose();
}

TEST(LegVelocityUpdate, GateRejectsASlippingFoot) {
  Rig rig;
  const Eigen::Vector3d v_true(0.3, 0.0, 0.0);
  for (int k = 1; k <= 20; k++)
    rig.step(0.5 * k, v_true);
  // LF slides 0.5 m/s backwards: its implied body velocity is off by 0.5 m/s
  ASSERT_TRUE(rig.step(10.5, v_true, 0, Eigen::Vector3d(-0.5, 0.0, 0.0)));
  EXPECT_EQ(rig.leg->last_accepted(), 1);
  EXPECT_EQ(rig.leg->last_rejected(), 1);
  EXPECT_LT((rig.state->imu->vel() - v_true.cast<DataType>()).norm(), 0.02);
}

TEST(LegVelocityUpdate, GyroBiasEntersThroughTheLeverArm) {
  // The rotation term [f]x (w_m - b_g): with the feet ~0.4 m out, a gyro
  // reading the state does not explain moves the residual by f x w
  Rig rig;
  const Eigen::Vector3d v_true(0.2, 0.0, 0.0);
  for (int k = 1; k <= 10; k++)
    rig.step(0.5 * k, v_true);
  rig.w_true = Vec3(0.0, 0.0, 0.5);
  rig.step(5.5, v_true);
  ASSERT_EQ(rig.leg->last_residuals().size(), 2u);
  for (const auto &r : rig.leg->last_residuals())
    EXPECT_GT(r.norm(), 0.1) << "an unexplained 0.5 rad/s yaw rate must show";
}

TEST(LegVelocityUpdate, NoJointDataNoUpdate) {
  Rig rig;
  rig.state->timestamp = 1.0;
  EXPECT_FALSE(rig.leg->try_update(rig.state, 1.0));
  // stale data: the newest sample is older than max_age_s
  for (int k = 0; k < 20; k++)
    rig.leg->feed_joint_state(rig.sample(0.2 + 0.005 * k, Eigen::Vector3d::Zero(), true), -1);
  rig.state->timestamp = 2.0;
  EXPECT_FALSE(rig.leg->try_update(rig.state, 2.0));
}

TEST(LegVelocityUpdate, ConvergesInTheWorldFrameUnderATiltedAttitude) {
  // The legs measure the velocity in the IMU frame; the state carries it in G.
  // With a yawed and tilted IMU the update must still recover v_IinG.
  Rig rig;
  const Mat3 R_GtoI =
      ov_core::rot_x<DataType>(0.2) * ov_core::rot_z<DataType>(0.7);
  VecX x = rig.state->imu->value();
  x.head(4) = ov_core::rot_2_quat(R_GtoI);
  rig.state->imu->set_value(x);
  rig.state->imu->set_fej(x);
  const Vec3 v_G(0.25, 0.1, 0.0);
  const Eigen::Vector3d v_B = (R_GtoI * v_G).cast<double>(); // body = IMU here
  for (int k = 1; k <= 20; k++)
    rig.step(0.5 * k, v_B);
  EXPECT_LT((rig.state->imu->vel() - v_G).norm(), 0.01)
      << "v = " << rig.state->imu->vel().transpose();
}

TEST(LegVelocityUpdate, RecoveryEscapesTheLockOut) {
  // A converged filter whose velocity jumps by 1 m/s while staying confident
  // (what a visual break does): every leg fails the gate. Without recovery the
  // legs stay locked out; with it the velocity is re-acquired.
  for (double recover_after : {0.0, 1.0}) {
    Rig rig(0.05, recover_after);
    const Eigen::Vector3d v_true(0.3, 0.0, 0.0);
    for (int k = 1; k <= 20; k++)
      rig.step(0.5 * k, v_true);
    VecX x = rig.state->imu->value();
    x.segment<3>(7) += Vec3(1.0, 0.0, 0.0); // q(4) p(3) v(3) bg ba
    rig.state->imu->set_value(x);
    for (int k = 21; k <= 40; k++)
      rig.step(0.5 * k, v_true);
    const DataType err = (rig.state->imu->vel() - v_true.cast<DataType>()).norm();
    if (recover_after > 0)
      EXPECT_LT(err, 0.02) << "recovered velocity error";
    else
      EXPECT_GT(err, 0.9) << "without recovery the legs stay locked out";
  }
}
