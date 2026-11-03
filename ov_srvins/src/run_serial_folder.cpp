/*
 * Square-root VINS
 * Copyright (C) 2025 Patrick Geneva
 * Copyright (C) 2025 Yuxiang Peng
 * Copyright (C) 2025 Guoquan Huang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 */

// Deterministic offline runner: replays a stereo + IMU folder (exported from a
// mission bag) through VioManager in timestamp order, with the same image
// throttling and camera queue logic as ROS2Visualizer, and no dropped frames.
//
// Folder layout (written by scripts/sparse_vio/export_mission.py in the
// lunarleaper-state-estimation-evaluation repo):
//   imu.csv     t_ns,wx,wy,wz,ax,ay,az
//   frames.csv  t_ns,cam0_png,cam1_png   (paths relative to the folder)
//
// Outputs in <out_dir>:
//   traj_odom.txt    TUM poses of the IMU in global, fast-propagated on the IMU
//                    clock at --odom_rate (what /odomimu publishes)
//   traj_update.txt  TUM poses after each processed frame (like /poseimu)
//   frames.csv       one row per frame that passed the track_frequency throttle
//   tracks.csv       tracked features after each frame (--dump_tracks 1)

#include <time.h>

#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "track/TrackBase.h"
#include "utils/colors.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_srvins;

namespace {

struct FrameRow {
  int64_t t_ns;
  std::string png0, png1;
};

double cpu_now() {
  timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double wall_now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ','))
    out.push_back(item);
  return out;
}

void write_tum(std::ofstream &of, double t,
               const Eigen::Matrix<DataType, 4, 1> &q,
               const Eigen::Matrix<DataType, 3, 1> &p) {
  // Same quaternion components as ROS2Visualizer puts in its pose messages
  of <<std::fixed << std::setprecision(9) << t << " " << std::setprecision(6)
     << p(0) << " " << p(1) << " " << p(2) << " " << q(0) << " " << q(1)
     << " " << q(2) << " " << q(3) << "\n";
}

} // namespace

int main(int argc, char **argv) {

  if (argc < 4) {
    PRINT_ERROR(RED "usage: run_serial_folder <config.yaml> <data_dir> "
                    "<out_dir> [--odom_rate 20] [--dump_tracks 1] "
                    "[--max_time_s -1]\n" RESET);
    return EXIT_FAILURE;
  }
  std::string config_path = argv[1];
  boost::filesystem::path data_dir(argv[2]);
  boost::filesystem::path out_dir(argv[3]);
  double odom_rate = 20.0;
  bool dump_tracks = true;
  double max_time_s = -1;
  for (int i = 4; i + 1 < argc; i += 2) {
    std::string key = argv[i];
    if (key == "--odom_rate")
      odom_rate = std::stod(argv[i + 1]);
    else if (key == "--dump_tracks")
      dump_tracks = std::stoi(argv[i + 1]) != 0;
    else if (key == "--max_time_s")
      max_time_s = std::stod(argv[i + 1]);
    else {
      PRINT_ERROR(RED "unknown option %s\n" RESET, key.c_str());
      return EXIT_FAILURE;
    }
  }
  boost::filesystem::create_directories(out_dir);

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);
  VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = false;
  params.use_multi_threading_pubs = false;
  if (!parser->successful()) {
    PRINT_ERROR(RED "[SERIAL]: unable to parse all parameters\n" RESET);
    return EXIT_FAILURE;
  }
  if (params.state_options.num_cameras != 2) {
    PRINT_ERROR(RED "[SERIAL]: only stereo folders are supported\n" RESET);
    return EXIT_FAILURE;
  }
  auto sys = std::make_shared<VioManager>(params);

  // Load the IMU and the frame list
  std::vector<ov_core::ImuData> imu;
  {
    std::ifstream f((data_dir / "imu.csv").string());
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
      auto v = split_csv(line);
      if (v.size() < 7)
        continue;
      ov_core::ImuData m;
      m.timestamp = std::stoll(v[0]) * 1e-9;
      m.wm << std::stod(v[1]), std::stod(v[2]), std::stod(v[3]);
      m.am << std::stod(v[4]), std::stod(v[5]), std::stod(v[6]);
      imu.push_back(m);
    }
  }
  std::vector<FrameRow> frames;
  {
    std::ifstream f((data_dir / "frames.csv").string());
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
      auto v = split_csv(line);
      if (v.size() < 3)
        continue;
      frames.push_back({std::stoll(v[0]), v[1], v[2]});
    }
  }
  if (imu.empty() || frames.empty()) {
    PRINT_ERROR(RED "[SERIAL]: no imu or frames in %s\n" RESET,
                data_dir.c_str());
    return EXIT_FAILURE;
  }
  PRINT_INFO("[SERIAL]: %zu imu, %zu stereo frames\n", imu.size(),
             frames.size());

  std::ofstream of_odom((out_dir / "traj_odom.txt").string());
  std::ofstream of_update((out_dir / "traj_update.txt").string());
  std::ofstream of_frames((out_dir / "frames.csv").string());
  std::ofstream of_tracks;
  of_odom << "# t x y z qx qy qz qw\n";
  of_update << "# t x y z qx qy qz qw\n";
  of_frames << "t_ns,wall_ms,cpu_ms,initialized,updated,clones,slam\n";
  if (dump_tracks) {
    of_tracks.open((out_dir / "tracks.csv").string());
    of_tracks << "t_ns,cam,id,u,v\n";
  }

  // Replay: images pass the same time throttle as ROS2Visualizer, wait in a
  // queue and are processed once an IMU sample is newer than them
  const double t_first = imu.front().timestamp;
  const double time_delta = 1.0 / params.track_frequency;
  double camera_last_timestamp = -1;
  double last_odom_t = -1;
  std::deque<ov_core::CameraData> camera_queue;
  std::deque<int64_t> queue_ns;
  size_t i_frame = 0;
  double cpu_total = 0, wall_start = wall_now();
  for (const auto &m : imu) {
    if (max_time_s > 0 && m.timestamp - t_first > max_time_s)
      break;

    // Images that were captured up to this IMU sample
    while (i_frame < frames.size() &&
           frames.at(i_frame).t_ns * 1e-9 <= m.timestamp) {
      const auto &fr = frames.at(i_frame++);
      double timestamp = fr.t_ns * 1e-9;
      if (camera_last_timestamp >= 0 &&
          timestamp < camera_last_timestamp + time_delta) {
        continue;
      }
      camera_last_timestamp = timestamp;
      ov_core::CameraData message;
      message.timestamp = timestamp;
      message.sensor_ids = {0, 1};
      for (const auto &png : {fr.png0, fr.png1}) {
        cv::Mat img =
            cv::imread((data_dir / png).string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
          PRINT_ERROR(RED "[SERIAL]: cannot read %s\n" RESET, png.c_str());
          return EXIT_FAILURE;
        }
        message.images.push_back(img);
      }
      for (int cam = 0; cam < 2; cam++) {
        if (params.use_mask) {
          message.masks.push_back(params.masks.at(cam));
        } else {
          message.masks.push_back(cv::Mat::zeros(message.images.at(cam).rows,
                                                 message.images.at(cam).cols,
                                                 CV_8UC1));
        }
      }
      camera_queue.push_back(message);
      queue_ns.push_back(fr.t_ns);
    }

    sys->feed_measurement_imu(m);

    double timestamp_imu_inC =
        m.timestamp - sys->get_state()->calib_dt_CAMtoIMU->value()(0);
    while (!camera_queue.empty() &&
           camera_queue.front().timestamp < timestamp_imu_inC) {
      double c0 = cpu_now(), w0 = wall_now();
      sys->feed_measurement_camera(camera_queue.front());
      double c1 = cpu_now(), w1 = wall_now();
      cpu_total += c1 - c0;
      auto state = sys->get_state();
      bool updated = sys->initialized() &&
                     state->timestamp == camera_queue.front().timestamp &&
                     state->clones_IMU.find(state->timestamp) !=
                         state->clones_IMU.end();
      of_frames << queue_ns.front() << "," << std::fixed
                << std::setprecision(3) << (w1 - w0) * 1e3 << ","
                << (c1 - c0) * 1e3 << "," << (int)sys->initialized() << ","
                << (int)updated << "," << state->clones_IMU.size() << ","
                << state->features_SLAM.size() << "\n";
      if (updated) {
        double t_ItoC = state->calib_dt_CAMtoIMU->value()(0);
        Eigen::Matrix<DataType, 4, 1> q = state->imu->quat();
        write_tum(of_update, state->timestamp + t_ItoC, q, state->imu->pos());
      }
      if (dump_tracks) {
        auto obs = sys->get_track_feat()->get_last_obs();
        auto ids = sys->get_track_feat()->get_last_ids();
        for (const auto &cam : obs) {
          const auto &id_cam = ids[cam.first];
          for (size_t k = 0; k < cam.second.size() && k < id_cam.size();
               k++) {
            of_tracks << queue_ns.front() << "," << cam.first << ","
                      << id_cam.at(k) << "," << std::setprecision(2)
                      << cam.second.at(k).pt.x << ","
                      << cam.second.at(k).pt.y << "\n";
          }
        }
      }
      camera_queue.pop_front();
      queue_ns.pop_front();
    }

    // Odometry on the IMU clock, like visualize_odometry()
    if (sys->initialized() &&
        (last_odom_t < 0 || m.timestamp >= last_odom_t + 1.0 / odom_rate)) {
      Eigen::Matrix<DataType, 13, 1> state_plus;
      Eigen::Matrix<DataType, 12, 12> cov_plus;
      if (sys->get_propagator()->fast_state_propagate(
              sys->get_state(), m.timestamp, state_plus, cov_plus)) {
        write_tum(of_odom, m.timestamp, state_plus.block(0, 0, 4, 1),
                  state_plus.block(4, 0, 3, 1));
        last_odom_t = m.timestamp;
      }
    }
  }

  PRINT_INFO("[SERIAL]: done, %.1f s wall, %.1f s cpu in the camera feed\n",
             wall_now() - wall_start, cpu_total);
  return EXIT_SUCCESS;
}
