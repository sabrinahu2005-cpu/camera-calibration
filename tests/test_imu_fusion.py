import numpy as np
import unittest

from websocket_xyzq_stream import RealTimeXyzqEstimator


class ImuFusionTests(unittest.TestCase):
    def test_estimator_accepts_enable_imu_fusion_flag(self):
        estimator = RealTimeXyzqEstimator(
            K_cam0=np.eye(3),
            K_cam2=np.eye(3),
            out_xyzq=__import__('pathlib').Path('tmp_xyzq.jsonl'),
            out_match=__import__('pathlib').Path('tmp_match.jsonl'),
            tolerance=0.03,
            vis_thresh=0.3,
            min_pairs=4,
            solve_batch_size=1,
            smooth_window=1,
            enable_imu_fusion=False,
        )
        self.assertFalse(estimator.enable_imu_fusion)
        estimator.add_imu_sample({
            'timestamp_wall': 1.0,
            'gx': 0.01,
            'gy': 0.0,
            'gz': 0.0,
            'ax': 0.0,
            'ay': 0.0,
            'az': 0.0,
        })
        estimator.close()


if __name__ == '__main__':
    unittest.main()
