import os, sys
sys.path.append("./hl2ss_")
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities


# Calibration path (must exist but can be empty)
calibration_path = 'calibration'
if not os.path.isdir(calibration_path):
    os.mkdir(calibration_path)


class Hl2Manager():
    def __init__(self, host, pv_width, pv_height, pv_fps):
        self.host = host
        self.pv_width = pv_width
        self.pv_height = pv_height

        # Buffer length in seconds
        buffer_size = 2

        # Start PV Subsystem ------------------------------------------------------
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO) #port : 3810

        # Get RM Depth AHAT calibration -------------------------------------------
        # Calibration data will be downloaded if it's not in the calibration folder
        self.calibration_ht = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_AHAT, calibration_path)

        uv2xy = self.calibration_ht.uv2xy  # hl2ss_3dcv.compute_uv2xy(calibration_ht.intrinsics, hl2ss.Parameters_RM_DEPTH_AHAT.WIDTH, hl2ss.Parameters_RM_DEPTH_AHAT.HEIGHT)
        xy1, self.scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, self.calibration_ht.scale)
        self.max_depth = self.calibration_ht.alias / self.calibration_ht.scale

        self.xy1_o = hl2ss_3dcv.block_to_list(xy1[:-1, :-1, :])
        self.xy1_d = hl2ss_3dcv.block_to_list(xy1[1:, 1:, :])

        # Start PV and RM Depth AHAT streams --------------------------------------
        self.producer = hl2ss_mp.producer()
        self.producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                           hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height,
                                           framerate=pv_fps))
        self.producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
        self.producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_fps * buffer_size)
        self.producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * buffer_size)
        self.producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        self.producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        self.sink_pv = consumer.create_sink(self.producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
        self.sink_ht = consumer.create_sink(self.producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)

        self.sink_pv.get_attach_response()
        self.sink_ht.get_attach_response()

        # Initialize PV intrinsics and extrinsics ---------------------------------
        self.pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
        self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)


    def receive_images(self, flag_depth):

        sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht, pv_width, pv_height = \
            self.sink_ht, self.sink_pv, self.pv_intrinsics, self.pv_extrinsics, self.xy1_o, self.xy1_d, self.scale, self.calibration_ht, self.pv_width, self.pv_height

        # Get RM Depth AHAT frame and nearest (in time) PV frame --------------
        _, data_ht = sink_ht.get_most_recent_frame()
        if ((data_ht is None) or (not hl2ss.is_valid_pose(data_ht.pose))):
            return None
        _, data_pv = sink_pv.get_nearest(data_ht.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            return None

        # Preprocess frames ---------------------------------------------------
        color = data_pv.payload.image

        pv_z = None
        if flag_depth:
            depth = data_ht.payload.depth  # hl2ss_3dcv.rm_depth_undistort(data_ht.payload.depth, calibration_ht.undistort_map)
            z = hl2ss_3dcv.rm_depth_normalize(depth, scale)

        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length,
                                                   data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        # Generate depth map for PV image -------------------------------------
        if flag_depth:
            mask = (depth[:-1, :-1].reshape((-1,)) > 0)
            zv = hl2ss_3dcv.block_to_list(z[:-1, :-1, :])[mask, :]

            ht_to_pv_image = hl2ss_3dcv.camera_to_rignode(calibration_ht.extrinsics) @ hl2ss_3dcv.reference_to_world(
                data_ht.pose) @ hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
                color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)

            ht_points_o = hl2ss_3dcv.rm_depth_to_points(xy1_o[mask, :], zv)
            pv_uv_o_h = hl2ss_3dcv.transform(ht_points_o, ht_to_pv_image)
            pv_list_depth = pv_uv_o_h[:, 2:]

            ht_points_d = hl2ss_3dcv.rm_depth_to_points(xy1_d[mask, :], zv)
            pv_uv_d_h = hl2ss_3dcv.transform(ht_points_d, ht_to_pv_image)
            pv_d_depth = pv_uv_d_h[:, 2:]

            mask = (pv_list_depth[:, 0] > 0) & (pv_d_depth[:, 0] > 0)

            pv_list_depth = pv_list_depth[mask, :]
            pv_d_depth = pv_d_depth[mask, :]

            pv_list_o = pv_uv_o_h[mask, 0:2] / pv_list_depth
            pv_list_d = pv_uv_d_h[mask, 0:2] / pv_d_depth

            pv_list = np.hstack((pv_list_o, pv_list_d + 1)).astype(np.int32)
            pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)

            u0 = pv_list[:, 0]
            v0 = pv_list[:, 1]
            u1 = pv_list[:, 2]
            v1 = pv_list[:, 3]

            mask0 = (u0 >= 0) & (u0 < pv_width) & (v0 >= 0) & (v0 < pv_height)
            mask1 = (u1 > 0) & (u1 <= pv_width) & (v1 > 0) & (v1 <= pv_height)
            maskf = mask0 & mask1

            pv_list = pv_list[maskf, :]
            pv_list_depth = pv_list_depth[maskf, 0]

            for n in range(0, pv_list.shape[0]):
                u0 = pv_list[n, 0]
                v0 = pv_list[n, 1]
                u1 = pv_list[n, 2]
                v1 = pv_list[n, 3]

                pv_z[v0:v1, u0:u1] = pv_list_depth[n]

        return color, pv_z

    def destroy(self):
        # Stop PV and RM Depth AHAT streams ---------------------------------------
        self.sink_pv.detach()
        self.sink_ht.detach()
        self.producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        self.producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)

        # Stop PV subsystem -------------------------------------------------------
        hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)