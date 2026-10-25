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

#include "LegKinematics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/core/persistence.hpp>

using namespace ov_srvins;

namespace {

/// Accepted closure gap for a leg to count as assembled [m]
constexpr double kAcceptGap = 1e-6;

template <typename T>
std::vector<T> read_seq(const cv::FileStorage &fs, const std::string &key) {
  std::vector<T> v;
  cv::FileNode node = fs[key];
  if (node.empty())
    throw std::runtime_error("leg mechanism: missing key " + key);
  if (node.isSeq() && node.size() > 0)
    node >> v;
  return v;
}

Eigen::Vector3d vec3(const std::vector<double> &v, size_t i) {
  return Eigen::Vector3d(v.at(3 * i), v.at(3 * i + 1), v.at(3 * i + 2));
}

Eigen::Matrix3d quat_wxyz(const std::vector<double> &v, size_t i) {
  return Eigen::Quaterniond(v.at(4 * i), v.at(4 * i + 1), v.at(4 * i + 2),
                            v.at(4 * i + 3))
      .normalized()
      .toRotationMatrix();
}

} // namespace

LegKinematics::LegKinematics(const std::string &mechanism_path) {
  cv::FileStorage fs(mechanism_path, cv::FileStorage::READ);
  if (!fs.isOpened())
    throw std::runtime_error("leg mechanism: cannot open " + mechanism_path);

  fs["num_bodies"] >> num_bodies_;
  fs["num_joints"] >> num_joints_;
  parent_ = read_seq<int>(fs, "joint_parent");
  child_ = read_seq<int>(fs, "joint_child");
  const auto axis = read_seq<double>(fs, "joint_axis");
  const auto pos0 = read_seq<double>(fs, "joint_pos0");
  const auto quat0 = read_seq<double>(fs, "joint_quat0");
  const auto pos1 = read_seq<double>(fs, "joint_pos1");
  const auto quat1 = read_seq<double>(fs, "joint_quat1");
  const size_t nj = (size_t)num_joints_;
  if (num_joints_ <= 0 || parent_.size() != nj || child_.size() != nj ||
      axis.size() != 3 * nj || pos0.size() != 3 * nj ||
      quat0.size() != 4 * nj || pos1.size() != 3 * nj ||
      quat1.size() != 4 * nj)
    throw std::runtime_error("leg mechanism: inconsistent joint arrays in " +
                             mechanism_path);
  for (size_t j = 0; j < nj; j++) {
    if (parent_[j] < 0 || parent_[j] >= num_bodies_ || child_[j] <= 0 ||
        child_[j] >= num_bodies_)
      throw std::runtime_error("leg mechanism: joint body index out of range");
    const V3 a = vec3(axis, j);
    revolute_.push_back(a.norm() > 0.5);
    axis_.push_back(revolute_.back() ? V3(a.normalized()) : V3::Zero());
    p0_.push_back(vec3(pos0, j));
    R0_.push_back(quat_wxyz(quat0, j));
    const M3 R1 = quat_wxyz(quat1, j);
    R1inv_.push_back(R1.transpose());
    p1inv_.push_back(-R1.transpose() * vec3(pos1, j));
  }

  // joints on the path root -> body; parents first means one pass suffices
  ancestor_.assign(num_bodies_, std::vector<bool>(nj, false));
  for (size_t j = 0; j < nj; j++) {
    ancestor_[child_[j]] = ancestor_[parent_[j]];
    ancestor_[child_[j]][j] = true;
  }

  cA_ = read_seq<int>(fs, "closure_body_a");
  cB_ = read_seq<int>(fs, "closure_body_b");
  const auto posA = read_seq<double>(fs, "closure_pos_a");
  const auto posB = read_seq<double>(fs, "closure_pos_b");
  for (size_t k = 0; k < cA_.size(); k++) {
    pA_.push_back(vec3(posA, k));
    pB_.push_back(vec3(posB, k));
  }

  const auto motors = read_seq<int>(fs, "motor_joints");
  motor_names_ = read_seq<std::string>(fs, "motor_names");
  const auto closures = read_seq<int>(fs, "leg_closures");
  const auto passives = read_seq<int>(fs, "leg_passives");
  const auto tl = read_seq<int>(fs, "leg_tl_body");
  const auto foot = read_seq<double>(fs, "foot_offset");
  fs["foot_radius"] >> foot_radius_;
  if (motors.size() != kMotors || motor_names_.size() != kMotors ||
      closures.size() != 2 * kLegs || passives.size() != 6 * kLegs ||
      tl.size() != kLegs || foot.size() != 3 * kLegs ||
      cA_.size() != cB_.size() || pA_.size() != cA_.size())
    throw std::runtime_error("leg mechanism: expected 4 legs x (3 motors, 2 "
                             "closures, 6 passives) in " +
                             mechanism_path);
  for (int leg = 0; leg < kLegs; leg++) {
    for (int m = 0; m < 3; m++)
      motor_joint_[leg][m] = motors[3 * leg + m];
    for (int c = 0; c < 2; c++)
      leg_closures_[leg][c] = closures[2 * leg + c];
    for (int p = 0; p < 6; p++)
      leg_passives_[leg][p] = passives[6 * leg + p];
    tl_body_[leg] = tl[leg];
    foot_offset_[leg] = vec3(foot, leg);
  }
  q_good_ = Eigen::VectorXd::Zero(num_joints_);
}

void LegKinematics::forward(const Eigen::VectorXd &q, Poses &poses) const {
  poses.body_R.resize(num_bodies_);
  poses.body_p.resize(num_bodies_);
  poses.joint_axis.resize(num_joints_);
  poses.joint_p.resize(num_joints_);
  poses.body_R[0].setIdentity();
  poses.body_p[0].setZero();
  for (int j = 0; j < num_joints_; j++) {
    const M3 &Rp = poses.body_R[parent_[j]];
    const M3 Rframe = Rp * R0_[j];
    const V3 pframe = poses.body_p[parent_[j]] + Rp * p0_[j];
    M3 Rj = Rframe;
    if (revolute_[j])
      Rj = Rframe * Eigen::AngleAxisd(q(j), axis_[j]).toRotationMatrix();
    poses.body_R[child_[j]] = Rj * R1inv_[j];
    poses.body_p[child_[j]] = pframe + Rj * p1inv_[j];
    poses.joint_axis[j] = Rframe * axis_[j];
    poses.joint_p[j] = pframe;
  }
}

void LegKinematics::closure(int leg, const Poses &poses,
                            Eigen::Matrix<double, 6, 1> &r,
                            Eigen::MatrixXd *J) const {
  if (J != nullptr)
    J->setZero(6, num_joints_);
  for (int c = 0; c < 2; c++) {
    const int k = leg_closures_[leg][c];
    const V3 xA = poses.body_p[cA_[k]] + poses.body_R[cA_[k]] * pA_[k];
    const V3 xB = poses.body_p[cB_[k]] + poses.body_R[cB_[k]] * pB_[k];
    r.segment<3>(3 * c) = xA - xB;
    if (J == nullptr)
      continue;
    for (int j = 0; j < num_joints_; j++) {
      if (!revolute_[j])
        continue;
      const V3 &w = poses.joint_axis[j];
      V3 d = V3::Zero();
      if (ancestor_[cA_[k]][j])
        d += w.cross(xA - poses.joint_p[j]);
      if (ancestor_[cB_[k]][j])
        d -= w.cross(xB - poses.joint_p[j]);
      J->block<3, 1>(3 * c, j) = d;
    }
  }
}

double LegKinematics::newton(int leg, Eigen::VectorXd &q, Poses &poses) const {
  auto gap = [](const Eigen::Matrix<double, 6, 1> &r) {
    return std::max(r.head<3>().norm(), r.tail<3>().norm());
  };
  Eigen::Matrix<double, 6, 1> r;
  Eigen::MatrixXd J;
  closure(leg, poses, r, &J);
  double g = gap(r);
  Poses trial;
  for (int it = 0; it < max_iter && g > tol; it++) {
    Eigen::Matrix<double, 6, 6> Cp;
    for (int p = 0; p < 6; p++)
      Cp.col(p) = J.col(leg_passives_[leg][p]);
    const Eigen::Matrix<double, 6, 1> dq = Cp.fullPivLu().solve(-r);
    // backtracking line search on the largest gap
    bool improved = false;
    double step = 1.0;
    for (int ls = 0; ls < 8 && !improved; ls++, step *= 0.5) {
      Eigen::VectorXd q_try = q;
      for (int p = 0; p < 6; p++)
        q_try(leg_passives_[leg][p]) += step * dq(p);
      forward(q_try, trial);
      Eigen::Matrix<double, 6, 1> r_try;
      closure(leg, trial, r_try, nullptr);
      if (gap(r_try) < g) {
        q = q_try;
        std::swap(poses, trial);
        improved = true;
      }
    }
    if (!improved)
      break; // stalled
    closure(leg, poses, r, &J);
    g = gap(r);
  }
  return g;
}

bool LegKinematics::solve(const Eigen::Matrix<double, kMotors, 1> &q_motors,
                          Result &out) {
  Eigen::VectorXd q = q_good_;
  Poses poses;
  for (int leg = 0; leg < kLegs; leg++) {
    Eigen::Vector3d start, target;
    for (int m = 0; m < 3; m++) {
      start(m) = q_good_(motor_joint_[leg][m]);
      target(m) = q_motors(3 * leg + m);
    }
    const Eigen::Vector3d delta = target - start;
    const int n_steps = std::max(
        1, (int)std::ceil(delta.cwiseAbs().maxCoeff() / max_step - 1e-12));
    bool ok = true;
    for (int k = 1; k <= n_steps && ok; k++) {
      for (int m = 0; m < 3; m++)
        q(motor_joint_[leg][m]) = start(m) + delta(m) * k / n_steps;
      forward(q, poses);
      ok = newton(leg, q, poses) < kAcceptGap;
    }
    if (ok) {
      for (int m = 0; m < 3; m++)
        q_good_(motor_joint_[leg][m]) = target(m);
      for (int p = 0; p < 6; p++)
        q_good_(leg_passives_[leg][p]) = q(leg_passives_[leg][p]);
    } else {
      // hold the last consistent passives, let the motors through
      for (int p = 0; p < 6; p++)
        q(leg_passives_[leg][p]) = q_good_(leg_passives_[leg][p]);
    }
    for (int m = 0; m < 3; m++)
      q(motor_joint_[leg][m]) = target(m);
    out.converged[leg] = ok;
  }

  forward(q, poses);
  for (int leg = 0; leg < kLegs; leg++) {
    const int tl = tl_body_[leg];
    const V3 foot = poses.body_p[tl] + poses.body_R[tl] * foot_offset_[leg];
    Eigen::Matrix<double, 6, 1> r;
    Eigen::MatrixXd C;
    closure(leg, poses, r, &C);
    // geometric Jacobian of the foot point over the joints of the TL chain
    auto foot_col = [&](int j) -> V3 {
      if (!revolute_[j] || !ancestor_[tl][j])
        return V3::Zero();
      return poses.joint_axis[j].cross(foot - poses.joint_p[j]);
    };
    Eigen::Matrix<double, 6, 6> Cp;
    Eigen::Matrix<double, 6, 3> Ca;
    Eigen::Matrix<double, 3, 6> Fp;
    M3 Fa;
    for (int p = 0; p < 6; p++) {
      Cp.col(p) = C.col(leg_passives_[leg][p]);
      Fp.col(p) = foot_col(leg_passives_[leg][p]);
    }
    for (int m = 0; m < 3; m++) {
      Ca.col(m) = C.col(motor_joint_[leg][m]);
      Fa.col(m) = foot_col(motor_joint_[leg][m]);
    }
    // closures stay closed: C_p dp + C_a da = 0
    const Eigen::Matrix<double, 6, 3> dp_da = -Cp.fullPivLu().solve(Ca);
    out.foot[leg] = foot;
    out.jacobian[leg] = Fa + Fp * dp_da;
  }
  return out.converged[0] && out.converged[1] && out.converged[2] &&
         out.converged[3];
}
