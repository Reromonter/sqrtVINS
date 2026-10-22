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

#include "utils/ImagePreprocess.h"

#include <algorithm>
#include <cmath>

namespace ov_srvins {

bool resolve_downsample_scale(bool downsample_cameras, double downsample_scale,
                              double &scale) {
  if (!(downsample_scale > 0.0 && downsample_scale <= 1.0)) {
    return false;
  }
  if (!downsample_cameras) {
    scale = downsample_scale;
    return true;
  }
  // The legacy flag only agrees with an unset (1) or explicit half scale
  if (downsample_scale != 1.0 && downsample_scale != 0.5) {
    return false;
  }
  scale = 0.5;
  return true;
}

bool valid_gaussian_blur_ksize(int ksize) {
  return ksize >= 3 && ksize % 2 == 1;
}

cv::Size scaled_image_size(int width, int height, double scale) {
  return cv::Size(std::max(1, (int)std::lround(width * scale)),
                  std::max(1, (int)std::lround(height * scale)));
}

void scale_camera_intrinsics(VecX &cam_calib, int width, int height,
                             double scale) {
  cv::Size wh = scaled_image_size(width, height, scale);
  DataType sx = (DataType)wh.width / (DataType)width;
  DataType sy = (DataType)wh.height / (DataType)height;
  cam_calib(0) *= sx;
  cam_calib(1) *= sy;
  cam_calib(2) = (cam_calib(2) + (DataType)0.5) * sx - (DataType)0.5;
  cam_calib(3) = (cam_calib(3) + (DataType)0.5) * sy - (DataType)0.5;
}

void preprocess_camera_image(const cv::Mat &img, const cv::Mat &mask,
                             bool gaussian_blur, double scale,
                             cv::Mat &img_out, cv::Mat &mask_out,
                             int blur_ksize) {
  // NOTE: img_blur must not alias img before the blur, else it blurs in place
  cv::Mat img_blur;
  if (gaussian_blur) {
    cv::GaussianBlur(img, img_blur, cv::Size(blur_ksize, blur_ksize), 0);
  } else {
    img_blur = img;
  }
  if (scale == 1.0) {
    img_out = img_blur;
    mask_out = mask;
    return;
  }
  cv::Size wh = scaled_image_size(img.cols, img.rows, scale);
  cv::resize(img_blur, img_out, wh, 0.0, 0.0, cv::INTER_AREA);
  if (mask.empty()) {
    mask_out = mask;
  } else {
    cv::resize(mask, mask_out, wh, 0.0, 0.0, cv::INTER_NEAREST);
  }
}

} // namespace ov_srvins
