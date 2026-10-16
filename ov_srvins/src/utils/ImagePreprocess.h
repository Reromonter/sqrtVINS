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

#ifndef OV_SRVINS_IMAGE_PREPROCESS_H
#define OV_SRVINS_IMAGE_PREPROCESS_H

#include <opencv2/opencv.hpp>

#include "utils/DataType.h"

namespace ov_srvins {

/// Kernel size of the optional gaussian blur. With sigma 0 OpenCV uses the
/// fixed 5-tap binomial kernel [1 4 6 4 1] / 16, the same one cv::pyrDown uses.
constexpr int GAUSSIAN_BLUR_KSIZE = 5;

/**
 * @brief Resolve the resolution scale of the tracking images.
 *
 * `downsample_cameras: true` is the legacy way of asking for half resolution
 * and means the same as `downsample_scale: 0.5`.
 *
 * @param downsample_cameras If the legacy half-resolution flag is set
 * @param downsample_scale Requested resolution scale, 1 is full resolution
 * @param scale Resolved scale in (0, 1]
 * @return False if the scale is outside (0, 1], or if both options are set
 * and disagree
 */
bool resolve_downsample_scale(bool downsample_cameras, double downsample_scale,
                              double &scale);

/**
 * @brief Size of a width x height image once its resolution is scaled
 *
 * @param width Width of the input image
 * @param height Height of the input image
 * @param scale Resolution scale in (0, 1]
 * @return Rounded size, at least one pixel per side
 */
cv::Size scaled_image_size(int width, int height, double scale);

/**
 * @brief Scale fx, fy, cx, cy of a camera calibration to the downsampled image
 *
 * Uses the per-axis ratio of the rounded output size and OpenCV's pixel-centre
 * convention for cv::resize, u' = (u + 0.5) * s - 0.5. Distortion coefficients
 * are resolution independent and left untouched.
 *
 * @param cam_calib Calibration (fx, fy, cx, cy, d1...d4), scaled in place
 * @param width Width of the calibrated image
 * @param height Height of the calibrated image
 * @param scale Resolution scale in (0, 1]
 */
void scale_camera_intrinsics(VecX &cam_calib, int width, int height,
                             double scale);

/**
 * @brief Blur and/or downsample a camera image and its mask before tracking
 *
 * The gaussian runs first, at the input resolution, so with a scale of 0.5 the
 * pair splits up what cv::pyrDown used to do in one go. The image is
 * downsampled with INTER_AREA and the mask with INTER_NEAREST, so the mask
 * stays binary. The inputs are never written to: they can share their buffer
 * with the incoming ROS message.
 *
 * @param img Grayscale camera image
 * @param mask Mask of the image (255 = do not track), may be empty
 * @param gaussian_blur If the image should be blurred
 * @param scale Resolution scale in (0, 1]
 * @param img_out Preprocessed image
 * @param mask_out Mask at the resolution of img_out
 */
void preprocess_camera_image(const cv::Mat &img, const cv::Mat &mask,
                             bool gaussian_blur, double scale,
                             cv::Mat &img_out, cv::Mat &mask_out);

} // namespace ov_srvins

#endif // OV_SRVINS_IMAGE_PREPROCESS_H
