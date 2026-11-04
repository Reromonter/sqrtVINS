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





#ifndef OV_CORE_TRACK_BASE_H
#define OV_CORE_TRACK_BASE_H

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "track/MotionPrior.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace ov_core {

class Feature;
class CamBase;
class FeatureDatabase;

/**
 * @brief Visual feature tracking base class
 *
 * This is the base class for all our visual trackers.
 * The goal here is to provide a common interface so all underlying trackers can
 * simply hide away all the complexities. We have something called the "feature
 * database" which has all the tracking information inside of it. The user can
 * ask this database for features which can then be used in an MSCKF or
 * batch-based setting. The feature tracks store both the raw (distorted) and
 * undistorted/normalized values. Right now we just support two camera models,
 * see: undistort_point_brown() and undistort_point_fisheye().
 *
 * @m_class{m-note m-warning}
 *
 * @par A Note on Multi-Threading Support
 * There is some support for asynchronous multi-threaded feature tracking of
 * independent cameras. The key assumption during implementation is that the
 * user will not try to track on the same camera in parallel, and instead call
 * on different cameras. For example, if I have two cameras, I can either
 * sequentially call the feed function, or I spin each of these into separate
 * threads and wait for their return. The @ref currid is atomic to allow for
 * multiple threads to access it without issue and ensure that all features have
 * unique id values. We also have mutex for access for the calibration and
 * previous images and tracks (used during visualization). It should be noted
 * that if a thread calls visualization, it might hang or the feed thread might,
 * due to acquiring the mutex for that specific camera id / feed.
 *
 * This base class also handles most of the heavy lifting with the
 * visualization, but the sub-classes can override this and do their own logic
 * if they want (i.e. the TrackAruco has its own logic for visualization). This
 * visualization needs access to the prior images and their tracks, thus must
 * synchronise in the case of multi-threading. This shouldn't impact
 * performance, but high frequency visualization calls can negatively effect the
 * performance.
 */
class TrackBase {

public:
  /**
   * @brief Desired pre-processing image method.
   */
  enum HistogramMethod { NONE, HISTOGRAM, CLAHE };

  /**
   * @brief Outlier rejection of temporal matches.
   *
   * FUNDAMENTAL is the RANSAC fundamental matrix of upstream. It is close to
   * degenerate when nearly every point lies on the ground plane. PLANE rejects
   * matches far from the motion prior's prediction (needs a valid prior, else
   * FUNDAMENTAL). RIGIDITY keeps the stereo points whose pairwise 3D distances
   * stay the same; mono-only tracks then fall back to FUNDAMENTAL.
   */
  enum OutlierCheck { FUNDAMENTAL, PLANE, RIGIDITY };

  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in
   * it
   * @param numfeats number of features we want want to track (i.e. track 200
   * points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our
   * non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done
   * (histogram eq?)
   */
  TrackBase(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras,
            int numfeats, int numaruco, bool stereo,
            HistogramMethod histmethod);

  virtual ~TrackBase() {}

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  virtual void feed_new_camera(const CameraData &message) = 0;

  /**
   * @brief Shows features extracted in the last image
   * @param img_out image to which we will overlayed features on
   * @param r1,g1,b1 first color to draw in
   * @param r2,g2,b2 second color to draw in
   * @param overlay Text overlay to replace to normal "cam0" in the top left of
   * screen
   */
  virtual void display_active(cv::Mat &img_out, int r1, int g1, int b1, int r2,
                              int g2, int b2, std::string overlay = "");

  /**
   * @brief Shows a "trail" for each feature (i.e. its history)
   * @param img_out image to which we will overlayed features on
   * @param r1,g1,b1 first color to draw in
   * @param r2,g2,b2 second color to draw in
   * @param highlighted unique ids which we wish to highlight (e.g. slam feats)
   * @param overlay Text overlay to replace to normal "cam0" in the top left of
   * screen
   */
  virtual void display_history(cv::Mat &img_out, int r1, int g1, int b1, int r2,
                               int g2, int b2,
                               std::vector<size_t> highlighted = {},
                               std::string overlay = "");

  /**
   * @brief Get the feature database with all the track information
   * @return FeatureDatabase pointer that one can query for features
   */
  std::shared_ptr<FeatureDatabase> get_feature_database() { return database; }

  /**
   * @brief Changes the ID of an actively tracked feature to another one.
   *
   * This function can be helpfull if you detect a loop-closure with an old
   * frame. One could then change the id of an active feature to match the old
   * feature id!
   *
   * @param id_old Old id we want to change
   * @param id_new Id we want to change the old id to
   */
  void change_feat_id(size_t id_old, size_t id_new);

  /// Getter method for active features in the last frame (observations per
  /// camera)
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> get_last_obs() {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    return pts_last;
  }

  /// Getter method for active features in the last frame (ids per camera)
  std::unordered_map<size_t, std::vector<size_t>> get_last_ids() {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    return ids_last;
  }

  /// Getter method for number of active features
  int get_num_features() { return num_features; }

  /// Setter method for number of active features
  void set_num_features(int _num_features) { num_features = _num_features; }

  /**
   * @brief Motion prior for the next feed_new_camera() call
   *
   * The prior predicts where the last tracked features land in the next
   * image (see MotionPrior). It is used by the KLT start points (use_prior),
   * the PLANE outlier check and guided descriptor matching. An invalid prior
   * makes all of them fall back to upstream behaviour.
   */
  void set_motion_prior(const MotionPrior &prior) { motion_prior = prior; }

  /**
   * @brief Options for the motion prior and the outlier check
   * @param prior_starts If KLT should start from the predicted points
   * @param check Outlier rejection of temporal matches
   * @param plane_px PLANE: max distance to the prediction (tracking pixels)
   * @param rigid_abs_m,rigid_rel RIGIDITY: allowed distance change is
   * rigid_abs_m + rigid_rel * depth
   * @param guided_px Guided descriptor matching radius around the
   * prediction (tracking pixels), <= 0 is off
   */
  void set_prior_options(bool prior_starts, OutlierCheck check,
                         double plane_px, double rigid_abs_m, double rigid_rel,
                         double guided_px) {
    use_prior = prior_starts;
    outlier_check = check;
    plane_outlier_px = plane_px;
    rigidity_abs_m = rigid_abs_m;
    rigidity_rel = rigid_rel;
    guided_radius_px = guided_px;
  }

protected:
  /// Camera object which has all calibration in it
  std::unordered_map<size_t, std::shared_ptr<CamBase>> camera_calib;

  /// Database with all our current features
  std::shared_ptr<FeatureDatabase> database;

  /// If we are a fisheye model or not
  std::map<size_t, bool> camera_fisheye;

  /// Number of features we should try to track frame to frame
  int num_features;

  /// If we should use binocular tracking or stereo tracking for multi-camera
  bool use_stereo;

  /// What histogram equalization method we should pre-process images with?
  HistogramMethod histogram_method;

  /// Mutexs for our last set of image storage (img_last, pts_last, and
  /// ids_last)
  std::vector<std::mutex> mtx_feeds;

  /// Mutex for editing the *_last variables
  std::mutex mtx_last_vars;

  /// Last set of images (use map so all trackers render in the same order)
  std::map<size_t, cv::Mat> img_last;

  /// Last set of images (use map so all trackers render in the same order)
  std::map<size_t, cv::Mat> img_mask_last;

  /// Last set of tracked points
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> pts_last;

  /// Set of IDs of each current feature in the database
  std::unordered_map<size_t, std::vector<size_t>> ids_last;

  /// Master ID for this tracker (atomic to allow for multi-threading)
  std::atomic<size_t> currid;

  // Timing variables (most children use these...)
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  /// Motion prior for the next image and how it is used
  MotionPrior motion_prior;
  bool use_prior = false;
  OutlierCheck outlier_check = FUNDAMENTAL;
  double plane_outlier_px = 20.0;
  double rigidity_abs_m = 0.03;
  double rigidity_rel = 0.03;
  double guided_radius_px = -1.0;

  /// If the prior can be used for this image (valid and for its timestamp)
  bool prior_for(double timestamp) const {
    return motion_prior.valid &&
           std::abs(motion_prior.timestamp_new - timestamp) < 1e-6;
  }

  /**
   * @brief Predicts the new pixels of the old points of one camera
   *
   * Points whose id the other camera of the stereo pair also saw are
   * triangulated and projected with the full pose; the rest use the ground
   * plane / rotation of the prior. Points without a prediction keep their
   * old pixel and have has_pred false.
   */
  std::vector<cv::Point2f>
  predict_points(size_t cam_id, const std::vector<cv::KeyPoint> &pts,
                 const std::vector<size_t> &ids, int cam_other,
                 const std::vector<cv::KeyPoint> &pts_other,
                 const std::vector<size_t> &ids_other,
                 std::vector<char> &has_pred, int &num_stereo);

  /**
   * @brief Prints a [PRED] line: distance of the final tracks to the
   * prediction, split into the near field (lower two thirds of the image) and
   * the far field (top third)
   */
  void log_prediction_error(double timestamp, size_t cam_id,
                            const std::vector<cv::Point2f> &pred,
                            const std::vector<char> &has_pred,
                            const std::vector<cv::KeyPoint> &final_pts,
                            const std::vector<uchar> &ok, int rows,
                            int num_stereo);

  /**
   * @brief Rigidity check of stereo tracks between two stereo frames
   * @return Inlier flag per entry (old/new raw pixels of both cameras); all
   * true if the prior carries no camera models
   */
  std::vector<bool> rigidity_check(size_t cam0, size_t cam1,
                                   const std::vector<cv::Point2f> &uv0_old,
                                   const std::vector<cv::Point2f> &uv1_old,
                                   const std::vector<cv::Point2f> &uv0_new,
                                   const std::vector<cv::Point2f> &uv1_new);
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_BASE_H */
