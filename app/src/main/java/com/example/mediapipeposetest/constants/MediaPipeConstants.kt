package com.example.mediapipeposetest.constants

import com.google.mediapipe.components.CameraHelper

const val BINARY_GRAPH_NAME = "pose_tracking_gpu.binarypb"
const val ORIGINAL_VIDEO_STREAM = "input_video"
const val PROCESSED_VIDEO_STREAM = "output_video"
const val OUTPUT_LANDMARK_STREAM_NAME = "pose_landmarks"

val CAMERA_FACING_BACK: CameraHelper.CameraFacing = CameraHelper.CameraFacing.BACK
val CAMERA_FACING_FRONT: CameraHelper.CameraFacing = CameraHelper.CameraFacing.FRONT

const val FLIP_FRAMES_VERTICALLY = false
