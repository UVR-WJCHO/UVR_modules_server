import os, sys

sys.path.append("./_hl2ss")
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities
import multiprocessing as mp
import numpy as np


# Calibration path (must exist but can be empty)
CALIBRATION_PATH = '_calibration'
if not os.path.isdir(CALIBRATION_PATH):
    os.mkdir(CALIBRATION_PATH)


class Hl2Manager:
    """Manages communication and data stream from HoloLens 2 using hl2ss."""

    def __init__(self, host, pv_width, pv_height, pv_fps, buffer_size=10):
        self.host = host
        self.pv_width = pv_width
        self.pv_height = pv_height
        self.pv_fps = pv_fps
        self.buffer_size = buffer_size

        # 1. Start PV Subsystem
        hl2ss_lnm.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # 2. Get RM Depth AHAT calibration
        self.calibration_ht = hl2ss_3dcv.get_calibration_rm(
            self.host, hl2ss.StreamPort.RM_DEPTH_AHAT, CALIBRATION_PATH
        )

        uv2xy = self.calibration_ht.uv2xy
        xy1, self.scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, self.calibration_ht.scale)
        self.max_depth = self.calibration_ht.alias / self.calibration_ht.scale

        self.xy1_o = hl2ss_3dcv.block_to_list(xy1[:-1, :-1, :])
        self.xy1_d = hl2ss_3dcv.block_to_list(xy1[1:, 1:, :])

        # 3. Start PV and RM Depth AHAT streams
        self.producer = hl2ss_mp.producer()
        self.producer.configure(
            hl2ss.StreamPort.PERSONAL_VIDEO,
            hl2ss_lnm.rx_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height,
                            framerate=pv_fps)
        )
        self.producer.configure(
            hl2ss.StreamPort.RM_DEPTH_AHAT,
            hl2ss_lnm.rx_rm_depth_ahat(self.host, hl2ss.StreamPort.RM_DEPTH_AHAT)
        )
        self.producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_fps * buffer_size)
        self.producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * buffer_size)
        self.producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        self.producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

        # 4. Create consumers
        self.consumer = hl2ss_mp.consumer()
        self.manager = mp.Manager()
        self.sink_pv = self.consumer.create_sink(self.producer, hl2ss.StreamPort.PERSONAL_VIDEO, self.manager, None)
        self.sink_ht = self.consumer.create_sink(self.producer, hl2ss.StreamPort.RM_DEPTH_AHAT, self.manager, None)

        self.sink_pv.get_attach_response()
        self.sink_ht.get_attach_response()

        # 5. Initialize PV intrinsics and extrinsics
        self.pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
        self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    def receive_images(self, flag_depth):
        """
        Receives and processes image data from HL2, optionally including aligned depth map.

        Returns:
            tuple: (color_image, pv_aligned_depth_map) or None on failure.
        """
        # Get RM Depth AHAT frame and nearest (in time) PV frame
        _, data_ht = self.sink_ht.get_most_recent_frame()
        if ((data_ht is None) or (not hl2ss.is_valid_pose(data_ht.pose))):
            return None
        _, data_pv = self.sink_pv.get_nearest(data_ht.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            return None

        color = data_pv.payload.image
        pv_z = None

        if flag_depth:
            depth = data_ht.payload.depth
            z = hl2ss_3dcv.rm_depth_normalize(depth, self.scale)

            # Update PV intrinsics (may change due to autofocus)
            self.pv_intrinsics = hl2ss.update_pv_intrinsics(
                self.pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point
            )
            color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(
                self.pv_intrinsics, self.pv_extrinsics
            )

            # Generate depth map for PV image (PV-aligned depth map)
            mask = (depth[:-1, :-1].reshape((-1,)) > 0)
            zv = hl2ss_3dcv.block_to_list(z[:-1, :-1, :])[mask, :]

            # Compute transformation from HT camera to PV image plane
            ht_to_pv_image = hl2ss_3dcv.camera_to_rignode(self.calibration_ht.extrinsics) @ \
                             hl2ss_3dcv.reference_to_world(data_ht.pose) @ \
                             hl2ss_3dcv.world_to_reference(data_pv.pose) @ \
                             hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ \
                             hl2ss_3dcv.camera_to_image(color_intrinsics)

            ht_points_o = hl2ss_3dcv.rm_depth_to_points(self.xy1_o[mask, :], zv)
            pv_uv_o_h = hl2ss_3dcv.transform(ht_points_o, ht_to_pv_image)
            pv_list_depth = pv_uv_o_h[:, 2:]

            ht_points_d = hl2ss_3dcv.rm_depth_to_points(self.xy1_d[mask, :], zv)
            pv_uv_d_h = hl2ss_3dcv.transform(ht_points_d, ht_to_pv_image)
            pv_d_depth = pv_uv_d_h[:, 2:]

            mask_depth = (pv_list_depth[:, 0] > 0) & (pv_d_depth[:, 0] > 0)
            pv_list_depth = pv_list_depth[mask_depth, :]
            pv_d_depth = pv_d_depth[mask_depth, :]

            pv_list_o = pv_uv_o_h[mask_depth, 0:2] / pv_list_depth
            pv_list_d = pv_uv_d_h[mask_depth, 0:2] / pv_d_depth

            pv_list = np.hstack((pv_list_o, pv_list_d + 1)).astype(np.int32)
            pv_z = np.zeros((self.pv_height, self.pv_width), dtype=np.float32)

            u0, v0 = pv_list[:, 0], pv_list[:, 1]
            u1, v1 = pv_list[:, 2], pv_list[:, 3]

            mask0 = (u0 >= 0) & (u0 < self.pv_width) & (v0 >= 0) & (v0 < self.pv_height)
            mask1 = (u1 > 0) & (u1 <= self.pv_width) & (v1 > 0) & (v1 <= self.pv_height)
            maskf = mask0 & mask1

            pv_list = pv_list[maskf, :]
            pv_list_depth = pv_list_depth[maskf, 0]

            for n in range(0, pv_list.shape[0]):
                u0, v0 = pv_list[n, 0], pv_list[n, 1]
                u1, v1 = pv_list[n, 2], pv_list[n, 3]
                pv_z[v0:v1, u0:u1] = pv_list_depth[n]

        return color, pv_z, np.copy(self.pv_intrinsics)

    def destroy(self):
        """Stops all running hl2ss streams and subsystems."""
        # Stop PV and RM Depth AHAT streams
        self.sink_pv.detach()
        self.sink_ht.detach()
        self.producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        self.producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)

        # Stop PV subsystem
        hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)