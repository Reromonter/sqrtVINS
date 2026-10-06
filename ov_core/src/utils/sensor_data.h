/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
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





#ifndef OV_CORE_SENSOR_DATA_H
#define OV_CORE_SENSOR_DATA_H

#include "utils/DataType.h"
#include "utils/print.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ov_core {

/**
 * @brief Struct for a single imu measurement (time, wm, am)
 */
struct ImuData {

  /// Timestamp of the reading
  double timestamp;

  /// Gyroscope reading, angular velocity (rad/s)
  Eigen::Matrix<DataType, 3, 1> wm;

  /// Accelerometer reading, linear acceleration (m/s^2)
  Eigen::Matrix<DataType, 3, 1> am;

  /// Sort function to allow for using of STL containers
  bool operator<(const ImuData &other) const {
    return timestamp < other.timestamp;
  }
};

/**
 * @brief Struct for a single two-axis sun sensor measurement (time, alpha,
 * beta)
 *
 * The angle convention is the one the Needronix NXSS3 model in
 * lunarleaper-state-estimation uses, and the two must stay in lock step:
 *
 *   alpha = atan2(sx, sz)   rotation about the sensor +Y axis
 *   beta  = atan2(sy, sz)   rotation about the sensor +X axis
 *   s     = [tan(alpha), tan(beta), 1] / ||.||
 *
 * where s is the unit vector pointing *towards* the sun, expressed in the
 * sensor frame.
 */
struct SunSensorData {

  /// Timestamp of the reading
  double timestamp;

  /// Sensor-frame angle about the sensor +Y axis (rad)
  DataType alpha;

  /// Sensor-frame angle about the sensor +X axis (rad)
  DataType beta;

  /// Sort function to allow for using of STL containers
  bool operator<(const SunSensorData &other) const {
    return timestamp < other.timestamp;
  }
};

/**
 * @brief Struct for a collection of camera measurements.
 *
 * For each image we have a camera id and timestamp that it occured at.
 * If there are multiple cameras we will treat it as pair-wise stereo tracking.
 */
struct CameraData {

  /// Timestamp of the reading
  double timestamp;

  /// Camera ids for each of the images collected
  std::vector<int> sensor_ids;

  /// Raw image we have collected for each camera
  std::vector<cv::Mat> images;

  /// Tracking masks for each camera we have
  std::vector<cv::Mat> masks;

  /// Sort function to allow for using of STL containers
  bool operator<(const CameraData &other) const {
    if (timestamp == other.timestamp) {
      int id = *std::min_element(sensor_ids.begin(), sensor_ids.end());
      int id_other =
          *std::min_element(other.sensor_ids.begin(), other.sensor_ids.end());
      return id < id_other;
    } else {
      return timestamp < other.timestamp;
    }
  }
};

} // namespace ov_core

#endif // OV_CORE_SENSOR_DATA_H