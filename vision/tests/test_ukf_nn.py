"""
Smoke tests for the 3D UKF-NN state estimator.

Tests cover:
  1. UKF3D predict/update roundtrip (state converges to measurement)
  2. Partial observation (NaN z-dim leaves z-state unchanged)
  3. Feature extractor: shape, purity of build_features, commit_innovation
  4. Outlier gating fires on extreme measurement
  5. Tracker rejects detection without valid depth
  6. Training feature extraction shape and consistency

All tests use real numpy/scipy.  torch is stubbed only to satisfy
the import chain (see conftest.py).
"""
import numpy as np
import pytest

from opss.state.ukf_nn.ukf import UKF3D
from opss.state.ukf_nn.features import (
    FeatureExtractor3D,
    extract_training_features_3d,
)
from opss.state.ukf_nn import config as cfg


# ---------------------------------------------------------------------------
# UKF3D
# ---------------------------------------------------------------------------

class TestUKF3DPredict:
    """Predict step produces valid state and covariance."""

    def test_predict_shapes(self):
        ukf = UKF3D()
        x = np.array([1.0, 2.0, 3.0, 0.5, 0.3, -0.1])
        P = np.eye(6) * 0.1
        x_pred, P_pred, delta_a = ukf.predict(x, P)

        assert x_pred.shape == (6,)
        assert P_pred.shape == (6, 6)
        assert delta_a.shape == (3,)

    def test_predict_without_nn_is_cv_plus_gravity_plus_hover(self):
        """Without NN, delta_a=0 and dynamics are CV + gravity + hover."""
        ukf = UKF3D()
        dt = ukf.dt
        g = ukf.gravity
        a_h = ukf.a_hover

        x0 = np.array([0.0, 0.0, 10.0, 1.0, 0.0, 0.0])
        P0 = np.eye(6) * 1e-12  # near-zero covariance for deterministic test

        x_pred, _, delta_a = ukf.predict(x0, P0)

        # delta_a should be zero (no NN)
        np.testing.assert_array_equal(delta_a, np.zeros(3))

        # Semi-implicit Euler: v_next = v + dt*(g + a_hover), p_next = p + dt*v_next
        # With hover: g + a_hover = [0,0,-9.81] + [0,0,+9.81] = [0,0,0]
        v_expected = x0[3:] + dt * (g + a_h)
        p_expected = x0[:3] + dt * v_expected
        x_expected = np.concatenate([p_expected, v_expected])

        np.testing.assert_allclose(x_pred, x_expected, atol=1e-10)

    def test_covariance_stays_symmetric_positive(self):
        ukf = UKF3D()
        x = np.array([5.0, -3.0, 20.0, 2.0, -1.0, 0.5])
        P = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

        x_pred, P_pred, _ = ukf.predict(x, P)

        # Symmetric
        np.testing.assert_allclose(P_pred, P_pred.T, atol=1e-14)
        # Positive semi-definite
        eigvals = np.linalg.eigvalsh(P_pred)
        assert np.all(eigvals >= -1e-12), f"Negative eigenvalue: {eigvals}"


class TestUKF3DUpdate:
    """Update step with full and partial observations."""

    def _predict_then_update(self, z):
        ukf = UKF3D()
        x = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        P = np.eye(6) * 1.0
        x_pred, P_pred, _ = ukf.predict(x, P)
        return ukf.update(x_pred, P_pred, z)

    def test_full_observation_reduces_uncertainty(self):
        z = np.array([1.1, 2.1, 3.1])
        x_upd, P_upd, innov, S = self._predict_then_update(z)

        assert x_upd.shape == (6,)
        assert P_upd.shape == (6, 6)
        assert innov.shape == (3,)
        assert S.shape == (3, 3)
        # Position uncertainty should decrease
        assert np.trace(P_upd[:3, :3]) < 3.0  # started at 3.0

    def test_partial_observation_nan_z(self):
        """NaN in z-dim: z-position and z-velocity should be unaffected."""
        ukf = UKF3D()
        x = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        P = np.eye(6) * 0.5
        x_pred, P_pred, _ = ukf.predict(x, P)

        # Store pre-update z-state
        z_pos_before = x_pred[2]
        z_vel_before = x_pred[5]
        P_zz_before = P_pred[2, 2]

        z = np.array([1.05, 2.05, np.nan])  # z-dim missing
        x_upd, P_upd, innov, S = ukf.update(x_pred, P_pred, z)

        # z-position and z-velocity unchanged (no z-measurement)
        assert x_upd[2] == pytest.approx(z_pos_before, abs=1e-12)
        assert x_upd[5] == pytest.approx(z_vel_before, abs=1e-12)
        assert P_upd[2, 2] == pytest.approx(P_zz_before, abs=1e-12)

        # Innovation: z-dim should be NaN
        assert np.isnan(innov[2])
        assert not np.isnan(innov[0])
        assert not np.isnan(innov[1])

    def test_no_observation_returns_prediction(self):
        z = np.array([np.nan, np.nan, np.nan])
        x_upd, P_upd, innov, S = self._predict_then_update(z)

        # Should be unchanged from prediction
        assert np.all(np.isnan(innov))
        assert np.all(np.isnan(S))

    def test_outlier_inflates_S(self):
        """Extreme outlier should trigger adaptive R inflation."""
        ukf = UKF3D()
        x = np.zeros(6)
        P = np.eye(6) * 0.01

        x_pred, P_pred, _ = ukf.predict(x, P)

        # Normal measurement
        z_normal = x_pred[:3] + 0.01
        _, _, _, S_normal = ukf.update(x_pred.copy(), P_pred.copy(), z_normal)

        # Extreme outlier (100m away)
        z_outlier = x_pred[:3] + 100.0
        _, _, _, S_outlier = ukf.update(x_pred.copy(), P_pred.copy(), z_outlier)

        # S should be inflated for the outlier
        assert np.trace(S_outlier) > np.trace(S_normal)


# ---------------------------------------------------------------------------
# FeatureExtractor3D
# ---------------------------------------------------------------------------

class TestFeatureExtractor3D:

    def test_build_features_shape(self):
        fe = FeatureExtractor3D()
        x = np.array([1.0, 2.0, 3.0, 0.5, 0.3, -0.1])
        features = fe.build_features(x)
        assert features.shape == (cfg.FEAT_DIM,)
        assert cfg.FEAT_DIM == 15

    def test_build_features_is_pure(self):
        """build_features must not mutate internal state."""
        fe = FeatureExtractor3D()
        fe.prev_innovation = np.array([0.1, 0.2, 0.3])

        x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        fe.build_features(x)

        # prev_innovation should be unchanged
        np.testing.assert_array_equal(
            fe.prev_innovation, [0.1, 0.2, 0.3]
        )

    def test_commit_innovation_updates_state(self):
        fe = FeatureExtractor3D()
        innov = np.array([0.5, -0.3, 0.1])
        fe.commit_innovation(innov)
        np.testing.assert_array_equal(fe.prev_innovation, innov)

    def test_commit_innovation_handles_nan(self):
        fe = FeatureExtractor3D()
        innov = np.array([0.5, np.nan, 0.1])
        fe.commit_innovation(innov)
        np.testing.assert_array_equal(fe.prev_innovation, [0.5, 0.0, 0.1])

    def test_feature_content(self):
        """Verify each feature slot contains the expected value."""
        fe = FeatureExtractor3D()
        fe.prev_innovation = np.array([0.1, -0.2, 0.3])
        fe.set_v_des(np.array([5.0, 6.0, 1.0]))

        x = np.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
        features = fe.build_features(x)
        speed = 5.0  # sqrt(9 + 16)

        # Axis 0 (x): base=0, v_error_x = 5.0 - 3.0 = 2.0
        assert features[0] == pytest.approx(3.0)    # v_x
        assert features[1] == pytest.approx(0.1)    # prev_innov_x
        assert features[2] == pytest.approx(0.1)    # |prev_innov_x|
        assert features[3] == pytest.approx(speed)  # ||v||
        assert features[4] == pytest.approx(2.0)    # v_error_x

        # Axis 1 (y): base=5, v_error_y = 6.0 - 4.0 = 2.0
        assert features[5] == pytest.approx(4.0)    # v_y
        assert features[6] == pytest.approx(-0.2)   # prev_innov_y
        assert features[7] == pytest.approx(0.2)    # |prev_innov_y|
        assert features[8] == pytest.approx(speed)
        assert features[9] == pytest.approx(2.0)    # v_error_y

        # Axis 2 (z): base=10, v_error_z = 1.0 - 0.0 = 1.0
        assert features[10] == pytest.approx(0.0)   # v_z
        assert features[11] == pytest.approx(0.3)   # prev_innov_z
        assert features[12] == pytest.approx(0.3)   # |prev_innov_z|
        assert features[13] == pytest.approx(speed)
        assert features[14] == pytest.approx(1.0)   # v_error_z

    def test_reset(self):
        fe = FeatureExtractor3D()
        fe.commit_innovation(np.array([1.0, 2.0, 3.0]))
        fe.set_v_des(np.array([4.0, 5.0, 6.0]))
        fe.reset()
        np.testing.assert_array_equal(fe.prev_innovation, np.zeros(3))
        np.testing.assert_array_equal(fe.v_des, np.zeros(3))


# ---------------------------------------------------------------------------
# Training feature extraction
# ---------------------------------------------------------------------------

class TestTrainingFeatures:

    def _make_trajectory(self, n=50):
        """Generate a simple parabolic trajectory."""
        dt = cfg.DT
        g = cfg.GRAVITY
        t = np.arange(n) * dt
        v0 = np.array([10.0, 5.0, 20.0])
        x0 = np.array([0.0, 0.0, 1.0])

        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        positions[0] = x0
        velocities[0] = v0

        for k in range(1, n):
            velocities[k] = velocities[k - 1] + dt * g
            positions[k] = positions[k - 1] + dt * velocities[k]

        return positions, velocities

    def test_output_shapes(self):
        pos, vel = self._make_trajectory(50)
        features, targets = extract_training_features_3d(pos, vel)
        assert features.shape == (48, cfg.FEAT_DIM)  # N-2
        assert targets.shape == (48, 3)

    def test_short_trajectory_returns_empty(self):
        pos = np.zeros((2, 3))
        vel = np.zeros((2, 3))
        features, targets = extract_training_features_3d(pos, vel)
        assert features.shape == (0, cfg.FEAT_DIM)
        assert targets.shape == (0, 3)

    def test_pure_gravity_targets_near_zero(self):
        """If trajectory is pure gravity, residual targets should be ~0.
        With hover compensation, a_phys = gravity + a_hover = [0,0,0] by default.
        For pure gravity trajectory (a_total = gravity), target = gravity - 0 = gravity.
        Pass a_phys=gravity explicitly for the original test semantics."""
        pos, vel = self._make_trajectory(50)
        features, targets = extract_training_features_3d(pos, vel, a_phys=cfg.GRAVITY)
        # Targets = (v[k+1]-v[k])/dt - gravity ≈ 0 for pure gravity
        np.testing.assert_allclose(targets, 0.0, atol=1e-10)

    def test_features_match_runtime_extractor(self):
        """Training features should match FeatureExtractor3D output."""
        pos, vel = self._make_trajectory(10)
        # Use gravity-only a_phys for consistent innovation computation
        a_phys = cfg.GRAVITY
        features, _ = extract_training_features_3d(pos, vel, a_phys=a_phys)

        # Manually compute what FeatureExtractor3D would produce
        dt = cfg.DT
        predicted_pos = pos[:-1] + dt * vel[:-1] + 0.5 * a_phys * dt ** 2
        innovations = pos[1:] - predicted_pos

        fe = FeatureExtractor3D()
        # For k=0 in training (which uses vel[1], prev_innov=innovations[0])
        # v_des defaults to zeros, v_error = -vel (same in both paths)
        fe.prev_innovation = innovations[0]
        runtime_feat = fe.build_features(
            np.concatenate([pos[1], vel[1]])
        )

        np.testing.assert_allclose(features[0], runtime_feat, atol=1e-12)


# ---------------------------------------------------------------------------
# Tracker depth rejection
# ---------------------------------------------------------------------------

class TestTrackerDepthRejection:

    def test_has_valid_depth(self):
        from opss.state.ukf_nn_tracker import UKFNNTracker

        assert UKFNNTracker._has_valid_depth({"depth": 10.0}) is True
        assert UKFNNTracker._has_valid_depth({"depth": 0.0}) is False
        assert UKFNNTracker._has_valid_depth({"depth": -1.0}) is False
        assert UKFNNTracker._has_valid_depth({}) is False

    def test_convert_detection_valid_depth(self):
        from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics

        tracker = object.__new__(UKFNNTracker)
        tracker.camera = CameraIntrinsics()
        tracker.R_world_from_cam = np.eye(3)
        tracker.t_world_from_cam = np.zeros(3)

        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 10.0}
        x_m, y_m, z_m = tracker._convert_detection(det)

        # At principal point, x_m and y_m should be ~0
        assert abs(x_m) < 0.01
        assert abs(y_m) < 0.01
        assert z_m == pytest.approx(10.0)

    def test_convert_detection_bounds_check(self):
        """Scene-radius invariant catches pixel-leak regressions."""
        from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics

        tracker = object.__new__(UKFNNTracker)
        # Rigged intrinsics: fx=1 makes x_m = (u-cx)*depth, so a big u
        # produces an absurd x_m, tripping the bounds check.
        tracker.camera = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        tracker.R_world_from_cam = np.eye(3)
        tracker.t_world_from_cam = np.zeros(3)

        det = {"center": {"x": 500.0, "y": 500.0}, "depth": 10.0}
        # x_m = 500*10/1 = 5000 → exceeds MAX_SCENE_RADIUS (200)
        with pytest.raises(AssertionError, match="scene bound"):
            tracker._convert_detection(det)

    def test_multi_tracker_skips_no_depth_detection(self):
        from opss.state.ukf_nn_tracker import MultiObjectUKFNN

        tracker = MultiObjectUKFNN()
        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 0.0}
        # Should not create any tracker
        tracker.update([det], timestamp=0.0)
        assert len(tracker.trackers) == 0

    def test_update_skips_when_depth_missing(self):
        """Matched detection with depth=0: update skips, misses increments."""
        from opss.state.ukf_nn_tracker import UKFNNTracker

        det_init = {"center": {"x": 320.0, "y": 240.0}, "depth": 5.0}
        t = UKFNNTracker(track_id=0, initial_detection=det_init, timestamp=0.0)

        # State before
        x_before = t.x.copy()
        misses_before = t.misses

        # Update with no depth — should skip
        det_no_depth = {"center": {"x": 325.0, "y": 245.0}, "depth": 0.0}
        t.update(det_no_depth, timestamp=0.1)

        assert t.misses == misses_before + 1
        # State unchanged (no UKF update happened)
        np.testing.assert_array_equal(t.x, x_before)


# ---------------------------------------------------------------------------
# Config consistency
# ---------------------------------------------------------------------------

class TestMetadataGuard:

    def test_stats_with_correct_metadata_passes(self):
        from opss.state.ukf_nn_tracker import _verify_stats_metadata

        stats = {
            "mean": [0.0] * cfg.FEAT_DIM,
            "std": [1.0] * cfg.FEAT_DIM,
            "feat_dim": cfg.FEAT_DIM,
        }
        _verify_stats_metadata(stats)  # should not raise

    def test_stats_with_wrong_feat_dim_raises(self):
        from opss.state.ukf_nn_tracker import _verify_stats_metadata

        stats = {
            "mean": [0.0] * 12,
            "std": [1.0] * 12,
            "feat_dim": 12,
        }
        with pytest.raises(RuntimeError, match="feat_dim=12"):
            _verify_stats_metadata(stats)

    def test_stats_with_wrong_length_no_metadata_raises(self):
        """Even without explicit feat_dim key, length mismatch is caught."""
        from opss.state.ukf_nn_tracker import _verify_stats_metadata

        stats = {
            "mean": [0.0] * 12,
            "std": [1.0] * 12,
        }
        with pytest.raises(RuntimeError, match="length 12"):
            _verify_stats_metadata(stats)

    def test_compute_stats_embeds_metadata(self):
        from opss.state.ukf_nn.features import compute_normalization_stats

        features = [np.random.randn(10, cfg.FEAT_DIM)]
        stats = compute_normalization_stats(features)
        assert stats["feat_dim"] == cfg.FEAT_DIM
        assert stats["nn_output_dim"] == cfg.NN_OUTPUT_DIM
        assert "version" in stats


class TestConfig:

    def test_dimensions_consistent(self):
        assert cfg.STATE_DIM == 6
        assert cfg.MEAS_DIM == 3
        assert cfg.ACCEL_DIM == 3
        assert cfg.FEAT_DIM == 15
        assert cfg.NN_INPUT_DIM == 15
        assert cfg.NN_OUTPUT_DIM == 3

    def test_q_r_shapes(self):
        assert cfg.Q.shape == (cfg.STATE_DIM, cfg.STATE_DIM)
        assert cfg.R.shape == (cfg.MEAS_DIM, cfg.MEAS_DIM)

    def test_gating_threshold(self):
        threshold = cfg.INNOVATION_THRESHOLD_SIGMA ** 2
        assert threshold == pytest.approx(25.0)

    def test_dt_is_exact(self):
        assert cfg.DT == pytest.approx(1.0 / 30.0)


# ---------------------------------------------------------------------------
# Camera extrinsics
# ---------------------------------------------------------------------------

class TestCameraExtrinsics:
    """Verify camera-to-world transform in the tracker pipeline."""

    def test_identity_extrinsics_matches_old_behavior(self):
        """With R=I, t=0, result is identical to raw pixel_to_meters."""
        from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics

        det = {"center": {"x": 350.0, "y": 260.0}, "depth": 8.0}
        cam = CameraIntrinsics()

        # Old behavior: pixel_to_meters directly
        x_old, y_old, z_old = cam.pixel_to_meters(350.0, 260.0, 8.0)

        # New behavior: identity extrinsics
        tracker = UKFNNTracker(
            track_id=0, initial_detection=det, timestamp=0.0,
            R_world_from_cam=np.eye(3), t_world_from_cam=np.zeros(3),
            camera=cam,
        )

        assert tracker.x[0] == pytest.approx(x_old, abs=1e-12)
        assert tracker.x[1] == pytest.approx(y_old, abs=1e-12)
        assert tracker.x[2] == pytest.approx(z_old, abs=1e-12)

    def test_known_rotation_camera_along_world_y(self):
        """
        Camera at world origin, looking along world +Y.
        Camera +Z (forward) = world +Y, camera +Y (down) = world -Z.

        R_world_from_cam columns = camera axes in world:
            cam_x → world +X:  [1, 0, 0]
            cam_y → world -Z:  [0, 0, -1]
            cam_z → world +Y:  [0, 1, 0]
        """
        from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics

        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ], dtype=np.float64)
        t = np.zeros(3)

        cam = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)

        # Detection at principal point, depth=10m
        # pixel_to_meters → camera-frame (0, 0, 10)
        # world = R @ (0,0,10) + 0 = (0, 10, 0) → world +Y at 10m
        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 10.0}
        tracker = UKFNNTracker(
            track_id=0, initial_detection=det, timestamp=0.0,
            R_world_from_cam=R, t_world_from_cam=t, camera=cam,
        )

        assert tracker.x[0] == pytest.approx(0.0, abs=1e-10)  # world x
        assert tracker.x[1] == pytest.approx(10.0, abs=1e-10)  # world y (depth)
        assert tracker.x[2] == pytest.approx(0.0, abs=1e-10)   # world z

    def test_known_rotation_with_translation(self):
        """Camera at world (10, -5, 10), looking along world +Y."""
        from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics

        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ], dtype=np.float64)
        t = np.array([10.0, -5.0, 10.0])

        cam = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)

        # Detection at principal point, depth=50m
        # camera-frame = (0, 0, 50)
        # world = R @ (0,0,50) + (10,-5,10) = (0,50,0) + (10,-5,10) = (10, 45, 10)
        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 50.0}
        tracker = UKFNNTracker(
            track_id=0, initial_detection=det, timestamp=0.0,
            R_world_from_cam=R, t_world_from_cam=t, camera=cam,
        )

        assert tracker.x[0] == pytest.approx(10.0, abs=1e-10)
        assert tracker.x[1] == pytest.approx(45.0, abs=1e-10)
        assert tracker.x[2] == pytest.approx(10.0, abs=1e-10)

    def test_get_pixel_position_round_trip(self):
        """pixel→world→pixel should recover original pixel coords."""
        from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics

        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ], dtype=np.float64)
        t = np.array([10.0, -5.0, 10.0])
        cam = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)

        det = {"center": {"x": 400.0, "y": 200.0}, "depth": 30.0}
        tracker = UKFNNTracker(
            track_id=0, initial_detection=det, timestamp=0.0,
            R_world_from_cam=R, t_world_from_cam=t, camera=cam,
        )

        u_out, v_out = tracker.get_pixel_position()
        assert u_out == pytest.approx(400.0, abs=1e-8)
        assert v_out == pytest.approx(200.0, abs=1e-8)

    def test_multi_tracker_passes_extrinsics(self):
        """MultiObjectUKFNN threads extrinsics to child trackers."""
        from opss.state.ukf_nn_tracker import MultiObjectUKFNN, CameraIntrinsics

        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ], dtype=np.float64)
        t = np.array([10.0, -5.0, 10.0])
        cam = CameraIntrinsics()

        multi = MultiObjectUKFNN(
            camera=cam, R_world_from_cam=R, t_world_from_cam=t,
        )

        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 50.0}
        multi.update([det], timestamp=0.0)

        assert len(multi.trackers) == 1
        tracker = list(multi.trackers.values())[0]
        np.testing.assert_array_equal(tracker.R_world_from_cam, R)
        np.testing.assert_array_equal(tracker.t_world_from_cam, t)


# ---------------------------------------------------------------------------
# Sim projection round-trip
# ---------------------------------------------------------------------------

class TestSimProjection:
    """Verify world → detection → tracker recovers world position."""

    def test_noiseless_round_trip(self):
        """world_to_detection → _convert_detection should recover position."""
        from opss.sim.camera import SimCamera, look_at_camera
        from opss.sim.projection import world_to_detection
        from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics

        # Camera at (10, -5, 10) looking at (10, 45, 10)
        sim_cam = look_at_camera(
            position=np.array([10.0, -5.0, 10.0]),
            target=np.array([10.0, 45.0, 10.0]),
        )

        # Ground truth world position
        p_world = np.array([12.0, 30.0, 14.0])

        # Project to detection (no noise)
        det = world_to_detection(p_world, sim_cam, noise=None)
        assert det is not None

        # Build tracker with matching extrinsics
        cam_intrinsics = CameraIntrinsics(
            fx=sim_cam.fx, fy=sim_cam.fy,
            cx=sim_cam.cx, cy=sim_cam.cy,
        )
        tracker = UKFNNTracker(
            track_id=0, initial_detection=det, timestamp=0.0,
            R_world_from_cam=sim_cam.R_world_from_cam,
            t_world_from_cam=sim_cam.t_world_from_cam,
            camera=cam_intrinsics,
        )

        # Recovered position should match truth
        assert tracker.x[0] == pytest.approx(p_world[0], abs=1e-8)
        assert tracker.x[1] == pytest.approx(p_world[1], abs=1e-8)
        assert tracker.x[2] == pytest.approx(p_world[2], abs=1e-8)

    def test_behind_camera_returns_none(self):
        """Points behind camera should not produce detections."""
        from opss.sim.camera import look_at_camera
        from opss.sim.projection import world_to_detection

        sim_cam = look_at_camera(
            position=np.array([10.0, -5.0, 10.0]),
            target=np.array([10.0, 45.0, 10.0]),
        )

        # Point behind camera (world y < camera y)
        p_behind = np.array([10.0, -20.0, 10.0])
        det = world_to_detection(p_behind, sim_cam, noise=None)
        assert det is None

    def test_out_of_image_returns_none(self):
        """Points outside image bounds should not produce detections."""
        from opss.sim.camera import look_at_camera
        from opss.sim.projection import world_to_detection

        sim_cam = look_at_camera(
            position=np.array([10.0, -5.0, 10.0]),
            target=np.array([10.0, 45.0, 10.0]),
        )

        # Point far to the side (should be outside 640x480 image)
        p_side = np.array([500.0, 30.0, 10.0])
        det = world_to_detection(p_side, sim_cam, noise=None)
        assert det is None

    def test_noisy_detection_has_correct_keys(self):
        """Noisy detection dict has all required keys."""
        from opss.sim.camera import look_at_camera
        from opss.sim.projection import world_to_detection
        from opss.sim.observation import ObservationNoise

        sim_cam = look_at_camera(
            position=np.array([10.0, -5.0, 10.0]),
            target=np.array([10.0, 45.0, 10.0]),
        )
        noise = ObservationNoise(pixel_noise_std=2.0, depth_noise_std=0.1)
        rng = np.random.default_rng(42)

        p_world = np.array([12.0, 30.0, 14.0])
        det = world_to_detection(p_world, sim_cam, noise=noise, rng=rng)

        assert det is not None
        assert "center" in det
        assert "x" in det["center"]
        assert "y" in det["center"]
        assert "depth" in det
        assert "bbox" in det
        assert "confidence" in det

    def test_look_at_axes_orthonormal(self):
        """look_at_camera should produce a proper rotation matrix."""
        from opss.sim.camera import look_at_camera

        cam = look_at_camera(
            position=np.array([5.0, -10.0, 3.0]),
            target=np.array([10.0, 40.0, 15.0]),
        )
        R = cam.R_world_from_cam

        # R should be orthogonal: R @ R.T = I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        # det(R) = 1 (proper rotation, not reflection)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Unit tagging
# ---------------------------------------------------------------------------

class TestUnitTagging:

    def test_ukf_nn_tracker_state_has_meters(self):
        """UKFNNTracker.get_state() must set units='meters'."""
        from opss.state.ukf_nn_tracker import UKFNNTracker

        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 5.0}
        t = UKFNNTracker(track_id=0, initial_detection=det, timestamp=0.0)
        state = t.get_state()
        assert state.units == "meters"

    def test_kalman_tracker_state_has_pixels(self):
        """KalmanTracker.get_state() must set units='pixels'."""
        from opss.state.kalman import KalmanTracker

        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 5.0}
        t = KalmanTracker(track_id=0, initial_detection=det, timestamp=0.0)
        state = t.get_state()
        assert state.units == "pixels"

    def test_object_state_default_units(self):
        """ObjectState default units should be 'pixels'."""
        from opss.state.kalman import ObjectState
        state = ObjectState(track_id=0, timestamp=0.0, x=0, y=0)
        assert state.units == "pixels"

    def test_object_state_to_dict_includes_units(self):
        from opss.state.kalman import ObjectState
        state = ObjectState(track_id=0, timestamp=0.0, x=0, y=0, units="meters")
        d = state.to_dict()
        assert d["units"] == "meters"


# ---------------------------------------------------------------------------
# Track limit enforcement
# ---------------------------------------------------------------------------

class TestMaxTracks:

    def test_ukf_nn_max_tracks_enforced(self):
        """MultiObjectUKFNN must not exceed max_tracks."""
        from opss.state.ukf_nn_tracker import MultiObjectUKFNN

        max_t = 3
        tracker = MultiObjectUKFNN(max_tracks=max_t)

        # Create more detections than max_tracks
        dets = [
            {"center": {"x": float(i * 50), "y": 240.0}, "depth": 5.0}
            for i in range(max_t + 5)
        ]
        tracker.update(dets, timestamp=0.0)
        assert len(tracker.trackers) == max_t

    def test_kalman_max_tracks_enforced(self):
        """MultiObjectKalmanFilter must not exceed max_tracks."""
        from opss.state.kalman import MultiObjectKalmanFilter

        max_t = 3
        tracker = MultiObjectKalmanFilter(max_tracks=max_t)

        dets = [
            {"center": {"x": float(i * 50), "y": 240.0}, "depth": 5.0}
            for i in range(max_t + 5)
        ]
        tracker.update(dets, timestamp=0.0)
        assert len(tracker.trackers) == max_t

    def test_ukf_nn_default_max_tracks(self):
        """Default max_tracks should match config."""
        from opss.state.ukf_nn_tracker import MultiObjectUKFNN
        tracker = MultiObjectUKFNN()
        assert tracker.max_tracks == cfg.MAX_TRACKS


# ---------------------------------------------------------------------------
# Config additions
# ---------------------------------------------------------------------------

class TestConfigAdditions:

    def test_max_tracks_exists(self):
        assert hasattr(cfg, "MAX_TRACKS")
        assert cfg.MAX_TRACKS == 50

    def test_intrinsic_warn_tolerance_exists(self):
        assert hasattr(cfg, "INTRINSIC_WARN_TOLERANCE")
        assert cfg.INTRINSIC_WARN_TOLERANCE == 0.05

    def test_p_max_diag_exists(self):
        assert hasattr(cfg, "P_MAX_DIAG")
        assert cfg.P_MAX_DIAG == 100.0


# ---------------------------------------------------------------------------
# Failure-mode recovery
# ---------------------------------------------------------------------------

class TestFailureModeRecovery:

    def test_nan_state_recovery(self):
        """Inject NaN into UKFNNTracker.x — predict() reverts to last-good."""
        from opss.state.ukf_nn_tracker import UKFNNTracker

        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 5.0}
        t = UKFNNTracker(track_id=0, initial_detection=det, timestamp=0.0)

        good_x = t.x.copy()

        # Inject NaN
        t.x[0] = np.nan
        t.predict(cfg.DT)

        # Should revert to last good state (the init state)
        assert np.all(np.isfinite(t.x))
        np.testing.assert_array_equal(t.x, good_x)

    def test_inf_covariance_recovery(self):
        """Inject inf into P — predict() reverts to last-good."""
        from opss.state.ukf_nn_tracker import UKFNNTracker

        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 5.0}
        t = UKFNNTracker(track_id=0, initial_detection=det, timestamp=0.0)

        good_x = t.x.copy()
        good_P = t.P.copy()

        # Inject inf into covariance
        t.P[0, 0] = np.inf
        t.predict(cfg.DT)

        # Should revert to last good state
        assert np.all(np.isfinite(t.x))
        assert np.all(np.isfinite(t.P))
        np.testing.assert_array_equal(t.x, good_x)
        np.testing.assert_array_equal(t.P, good_P)

    def test_kalman_singular_s_no_crash(self):
        """Force P=0 and R=0 in KalmanTracker — update() doesn't crash."""
        from opss.state.kalman import KalmanTracker

        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 5.0}
        t = KalmanTracker(track_id=0, initial_detection=det, timestamp=0.0)

        # Force singular S = H @ P @ H.T + R by zeroing both
        t.P = np.zeros((6, 6))
        t.R = np.zeros((3, 3))

        misses_before = t.misses
        # Should not crash
        t.update({"center": {"x": 325.0, "y": 245.0}, "depth": 5.1}, 0.1)
        assert t.misses == misses_before + 1

    def test_covariance_bounded_after_long_miss_streak(self):
        """100 consecutive predict-only steps — P stays within P_MAX_DIAG."""
        from opss.state.ukf_nn_tracker import UKFNNTracker

        det = {"center": {"x": 320.0, "y": 240.0}, "depth": 5.0}
        t = UKFNNTracker(track_id=0, initial_detection=det, timestamp=0.0)

        for _ in range(100):
            t.predict(cfg.DT)

        assert np.max(np.diag(t.P)) <= cfg.P_MAX_DIAG
