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

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "utils/ImagePreprocess.h"

using namespace ov_srvins;

namespace {

cv::Mat noise_image(int width, int height) {
  cv::Mat img(height, width, CV_8UC1);
  cv::RNG rng(42);
  rng.fill(img, cv::RNG::UNIFORM, 0, 256);
  return img;
}

/// Float image with a gaussian blob centred at pixel coordinates (u, v)
cv::Mat blob_image(int width, int height, double u, double v) {
  cv::Mat img(height, width, CV_32FC1);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double r2 = (x - u) * (x - u) + (y - v) * (y - v);
      img.at<float>(y, x) = (float)std::exp(-r2 / (2.0 * 4.0 * 4.0));
    }
  }
  return img;
}

cv::Point2d centroid(const cv::Mat &img) {
  cv::Moments m = cv::moments(img);
  return cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
}

} // namespace

TEST(ResolveDownsampleScale, LegacyFlagMeansHalf) {
  double scale = -1.0;
  ASSERT_TRUE(resolve_downsample_scale(true, 1.0, scale));
  EXPECT_DOUBLE_EQ(scale, 0.5);
  ASSERT_TRUE(resolve_downsample_scale(true, 0.5, scale));
  EXPECT_DOUBLE_EQ(scale, 0.5);
}

TEST(ResolveDownsampleScale, ScaleWithoutLegacyFlag) {
  double scale = -1.0;
  ASSERT_TRUE(resolve_downsample_scale(false, 1.0, scale));
  EXPECT_DOUBLE_EQ(scale, 1.0);
  ASSERT_TRUE(resolve_downsample_scale(false, 0.75, scale));
  EXPECT_DOUBLE_EQ(scale, 0.75);
}

TEST(ResolveDownsampleScale, RejectsConflictsAndOutOfRange) {
  double scale = -1.0;
  EXPECT_FALSE(resolve_downsample_scale(true, 0.75, scale));
  EXPECT_FALSE(resolve_downsample_scale(false, 0.0, scale));
  EXPECT_FALSE(resolve_downsample_scale(false, -0.5, scale));
  EXPECT_FALSE(resolve_downsample_scale(false, 1.5, scale));
  EXPECT_FALSE(resolve_downsample_scale(
      false, std::numeric_limits<double>::quiet_NaN(), scale));
  EXPECT_DOUBLE_EQ(scale, -1.0);
}

TEST(ScaledImageSize, RoundsAndKeepsOnePixel) {
  EXPECT_EQ(scaled_image_size(640, 400, 1.0), cv::Size(640, 400));
  EXPECT_EQ(scaled_image_size(640, 400, 0.5), cv::Size(320, 200));
  EXPECT_EQ(scaled_image_size(640, 400, 0.75), cv::Size(480, 300));
  EXPECT_EQ(scaled_image_size(640, 400, 0.33), cv::Size(211, 132));
  EXPECT_EQ(scaled_image_size(640, 400, 0.001), cv::Size(1, 1));
}

TEST(ScaleCameraIntrinsics, HalfResolution) {
  VecX calib(8);
  calib << 300, 310, 319.5, 199.5, 0.1, -0.2, 0.01, 0.02;
  scale_camera_intrinsics(calib, 640, 400, 0.5);
  EXPECT_FLOAT_EQ(calib(0), 150);
  EXPECT_FLOAT_EQ(calib(1), 155);
  // The optical centre of the image stays the optical centre
  EXPECT_FLOAT_EQ(calib(2), 159.5);
  EXPECT_FLOAT_EQ(calib(3), 99.5);
  EXPECT_FLOAT_EQ(calib(4), 0.1);
  EXPECT_FLOAT_EQ(calib(7), 0.02);
}

TEST(ScaleCameraIntrinsics, MatchesResizedImage) {
  // A point imaged through the scaled calibration must land where cv::resize
  // moves it, for even and uneven scales
  const int width = 640, height = 400;
  const double fx = 300, fy = 310, cx = 322.3, cy = 197.1;
  const double x_n = 0.27, y_n = -0.16;
  const double u = fx * x_n + cx, v = fy * y_n + cy;
  cv::Mat img = blob_image(width, height, u, v);
  for (double scale : {0.75, 0.5, 0.33, 0.25}) {
    VecX calib(8);
    calib << fx, fy, cx, cy, 0, 0, 0, 0;
    scale_camera_intrinsics(calib, width, height, scale);
    cv::Mat img_out, mask_out;
    preprocess_camera_image(img, cv::Mat(), false, scale, img_out, mask_out);
    cv::Point2d c = centroid(img_out);
    EXPECT_NEAR(c.x, calib(0) * x_n + calib(2), 0.02) << "scale " << scale;
    EXPECT_NEAR(c.y, calib(1) * y_n + calib(3), 0.02) << "scale " << scale;
  }
}

TEST(PreprocessCameraImage, PassThrough) {
  cv::Mat img = noise_image(64, 40);
  cv::Mat mask = cv::Mat::zeros(40, 64, CV_8UC1);
  cv::Mat img_out, mask_out;
  preprocess_camera_image(img, mask, false, 1.0, img_out, mask_out);
  EXPECT_EQ(img_out.data, img.data);
  EXPECT_EQ(mask_out.data, mask.data);
}

TEST(PreprocessCameraImage, BlurOnlyKeepsResolution) {
  cv::Mat img = noise_image(64, 40);
  cv::Mat img_orig = img.clone();
  cv::Mat mask = cv::Mat::zeros(40, 64, CV_8UC1);
  cv::Mat img_out, mask_out;
  preprocess_camera_image(img, mask, true, 1.0, img_out, mask_out);
  EXPECT_EQ(img_out.size(), img.size());
  EXPECT_EQ(mask_out.size(), mask.size());
  EXPECT_GT(cv::norm(img_out, img, cv::NORM_INF), 0.0);
  // The input can share its buffer with the ROS message, it must stay as is
  EXPECT_EQ(cv::norm(img, img_orig, cv::NORM_INF), 0.0);
}

TEST(PreprocessCameraImage, BlurIsThePyrDownKernel) {
  // cv::pyrDown is the gaussian followed by keeping every second pixel
  cv::Mat img = noise_image(64, 40);
  cv::Mat img_blur, mask_out, img_pyr;
  preprocess_camera_image(img, cv::Mat(), true, 1.0, img_blur, mask_out);
  cv::pyrDown(img, img_pyr);
  cv::Mat img_even;
  cv::resize(img_blur, img_even, img_pyr.size(), 0.0, 0.0, cv::INTER_NEAREST);
  // Up to rounding of the 8 bit intermediate
  EXPECT_LE(cv::norm(img_even, img_pyr, cv::NORM_INF), 1.0);
}

TEST(PreprocessCameraImage, DownsampleWithoutBlurDiffersFromPyrDown) {
  cv::Mat img = noise_image(64, 40);
  cv::Mat img_out, mask_out, img_pyr;
  preprocess_camera_image(img, cv::Mat(), false, 0.5, img_out, mask_out);
  cv::pyrDown(img, img_pyr);
  ASSERT_EQ(img_out.size(), img_pyr.size());
  EXPECT_GT(cv::norm(img_out, img_pyr, cv::NORM_INF), 1.0);
}

TEST(PreprocessCameraImage, MaskFollowsImageAndStaysBinary) {
  cv::Mat img = noise_image(640, 400);
  cv::Mat mask = cv::Mat::zeros(400, 640, CV_8UC1);
  mask(cv::Rect(0, 0, 640, 133)).setTo(255);
  mask(cv::Rect(201, 250, 37, 61)).setTo(255);
  for (double scale : {0.75, 0.5, 0.33}) {
    cv::Mat img_out, mask_out;
    preprocess_camera_image(img, mask, true, scale, img_out, mask_out);
    EXPECT_EQ(img_out.size(), scaled_image_size(640, 400, scale));
    EXPECT_EQ(mask_out.size(), img_out.size());
    cv::Mat not_binary = (mask_out != 0) & (mask_out != 255);
    EXPECT_EQ(cv::countNonZero(not_binary), 0) << "scale " << scale;
    double masked = cv::countNonZero(mask_out) / (double)mask_out.total();
    double masked_in = cv::countNonZero(mask) / (double)mask.total();
    EXPECT_NEAR(masked, masked_in, 0.01) << "scale " << scale;
  }
}

TEST(PreprocessCameraImage, EmptyMaskStaysEmpty) {
  cv::Mat img = noise_image(64, 40);
  cv::Mat img_out, mask_out;
  preprocess_camera_image(img, cv::Mat(), false, 0.5, img_out, mask_out);
  EXPECT_EQ(img_out.size(), cv::Size(32, 20));
  EXPECT_TRUE(mask_out.empty());
}
