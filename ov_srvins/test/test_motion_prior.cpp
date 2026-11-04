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


#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include "cam/CamRadtan.h"
#include "track/MotionPrior.h"
#include "utils/quat_ops.h"

using namespace ov_core;

namespace {

/// 640x400 pinhole camera without distortion
std::shared_ptr<CamBase> pinhole() {
  auto cam = std::make_shared<CamRadtan>(640, 400);
  VecX calib = VecX::Zero(8);
  calib << 400, 400, 320, 200, 0, 0, 0, 0;
  cam->set_value(calib);
  return cam;
}

/// cam0 looks forward (IMU x) and 30 deg down, cam1 7.7 cm to its right; IMU
/// 0.47 m above the ground plane z = 0, both IMU frames z-up
MotionPrior walking_prior(const Eigen::Vector3d &step, double yaw_deg) {
  MotionPrior prior;
  // camera z = IMU x pitched down, camera x = -IMU y, camera y = down-ish
  double p = 30.0 * M_PI / 180.0;
  Eigen::Matrix3d R_CtoI;
  R_CtoI.col(2) = Eigen::Vector3d(std::cos(p), 0, -std::sin(p));
  R_CtoI.col(0) = Eigen::Vector3d(0, -1, 0);
  R_CtoI.col(1) = R_CtoI.col(2).cross(R_CtoI.col(0));
  for (size_t cam = 0; cam < 2; cam++) {
    prior.R_ItoC[cam] = R_CtoI.transpose();
    Eigen::Vector3d p_CinI(0, cam == 0 ? 0.0 : -0.077, 0);
    prior.p_IinC[cam] = -R_CtoI.transpose() * p_CinI;
    prior.calib[cam] = pinhole();
  }
  prior.valid = true;
  prior.p_IoldinG = Eigen::Vector3d(0, 0, 0.47);
  prior.p_InewinG = prior.p_IoldinG + step;
  Eigen::Matrix3d R_yaw =
      Eigen::AngleAxisd(yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ())
          .toRotationMatrix();
  prior.R_GtoInew = R_yaw.transpose();
  prior.has_plane = true;
  prior.n_G = Eigen::Vector3d::UnitZ();
  prior.d_G = 0.0;
  return prior;
}

/// Pixel of a global point in camera cam at the old or new time
bool project(const MotionPrior &prior, size_t cam, bool at_new,
             const Eigen::Vector3d &p_FinG, cv::Point2f &uv) {
  Eigen::Matrix3d R_GtoC;
  Eigen::Vector3d p_CinG;
  prior.camera_pose(cam, at_new, R_GtoC, p_CinG);
  Eigen::Vector3d p_C = R_GtoC * (p_FinG - p_CinG);
  if (p_C(2) < 0.05)
    return false;
  uv = prior.calib.at(cam)->distort(
      cv::Point2f((float)(p_C(0) / p_C(2)), (float)(p_C(1) / p_C(2))));
  return true;
}

} // namespace

TEST(MotionPrior, GroundPointsFollowThePlaneHomography) {
  MotionPrior prior = walking_prior(Eigen::Vector3d(0.3, 0.05, 0), 10.0);
  for (double x : {0.6, 1.0, 2.0, 4.0}) {
    for (double y : {-0.4, 0.0, 0.3}) {
      Eigen::Vector3d p_FinG(x, y, 0.0);
      cv::Point2f uv_old, uv_true, uv_pred;
      ASSERT_TRUE(project(prior, 0, false, p_FinG, uv_old));
      if (!project(prior, 0, true, p_FinG, uv_true))
        continue;
      ASSERT_TRUE(prior.predict(0, uv_old, nullptr, uv_pred));
      EXPECT_NEAR(uv_pred.x, uv_true.x, 0.05);
      EXPECT_NEAR(uv_pred.y, uv_true.y, 0.05);
    }
  }
}

TEST(MotionPrior, KnownDepthUsesTheFullPose) {
  // a point 0.3 m above the ground: the plane would be wrong, the 3D point
  // is right
  MotionPrior prior = walking_prior(Eigen::Vector3d(0.25, 0, 0), 0.0);
  Eigen::Vector3d p_FinG(1.5, 0.2, 0.3);
  cv::Point2f uv_old, uv_true, uv_pred, uv_plane;
  ASSERT_TRUE(project(prior, 0, false, p_FinG, uv_old));
  ASSERT_TRUE(project(prior, 0, true, p_FinG, uv_true));
  ASSERT_TRUE(prior.predict(0, uv_old, &p_FinG, uv_pred));
  ASSERT_TRUE(prior.predict(0, uv_old, nullptr, uv_plane));
  EXPECT_NEAR(cv::norm(uv_pred - uv_true), 0.0, 0.05);
  EXPECT_GT(cv::norm(uv_plane - uv_true), 5.0);
}

TEST(MotionPrior, StereoTriangulationRecoversThePoint) {
  MotionPrior prior = walking_prior(Eigen::Vector3d::Zero(), 0.0);
  Eigen::Vector3d p_FinG(1.2, -0.3, 0.0), p_est;
  cv::Point2f uv0, uv1;
  ASSERT_TRUE(project(prior, 0, false, p_FinG, uv0));
  ASSERT_TRUE(project(prior, 1, false, p_FinG, uv1));
  ASSERT_TRUE(prior.triangulate_stereo(0, 1, false, prior.undistort(0, uv0),
                                       prior.undistort(1, uv1), p_est));
  EXPECT_LT((p_est - p_FinG).norm(), 1e-3);
}

TEST(MotionPrior, GroundPlaneFitRejectsOutliersAndChecksTheNormal) {
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 40; i++)
    points.emplace_back(0.5 + 0.05 * i, std::sin(i) * 0.5,
                        -0.47 + 0.002 * std::cos(3 * i));
  for (int i = 0; i < 10; i++)
    points.emplace_back(1.0 + 0.1 * i, 0.2, -0.1 + 0.03 * i); // a rock
  Eigen::Vector3d n;
  double d;
  int inliers;
  ASSERT_TRUE(fit_ground_plane(points, Eigen::Vector3d::UnitZ(), 20.0, 0.03,
                               15, n, d, inliers));
  EXPECT_GE(inliers, 38);
  EXPECT_LT(std::acos(n.dot(Eigen::Vector3d::UnitZ())) * 180.0 / M_PI, 1.0);
  EXPECT_NEAR(d, -0.47, 0.01);
  // a wall is no ground
  std::vector<Eigen::Vector3d> wall;
  for (int i = 0; i < 40; i++)
    wall.emplace_back(2.0, 0.05 * i, 0.03 * (i % 7));
  EXPECT_FALSE(fit_ground_plane(wall, Eigen::Vector3d::UnitZ(), 20.0, 0.03, 15,
                                n, d, inliers));
}

TEST(MotionPrior, RigidityFlagsTheMovedPoints) {
  std::vector<Eigen::Vector3d> p_old, p_new;
  Eigen::Matrix3d R =
      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  Eigen::Vector3d t(0.1, 0.0, -0.3);
  for (int i = 0; i < 20; i++) {
    Eigen::Vector3d p(0.3 * std::sin(i), 0.2 * std::cos(2 * i), 1.0 + 0.1 * i);
    p_old.push_back(p);
    p_new.push_back(R * p + t);
  }
  // two outliers that moved by 0.3 m on their own, along the depth spread
  p_new.at(3) += Eigen::Vector3d(0, 0, 0.3);
  p_new.at(11) += Eigen::Vector3d(0, -0.3, 0.1);
  std::vector<bool> inl = rigidity_inliers(p_old, p_new, 0.03, 0.03);
  for (int i = 0; i < 20; i++)
    EXPECT_EQ(inl.at(i), i != 3 && i != 11) << "point " << i;

  // known limit: a move across the depth spread barely changes the distances
  // (12 of 19 stay in tolerance), so it passes the 0.5 share
  p_new.at(3) = R * p_old.at(3) + t + Eigen::Vector3d(0.3, 0, 0);
  inl = rigidity_inliers(p_old, p_new, 0.03, 0.03);
  EXPECT_TRUE(inl.at(3));
  EXPECT_FALSE(inl.at(11));
}
