/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.mediapipeposetest

import android.graphics.SurfaceTexture
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.*
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

import com.google.mediapipe.components.*
import com.google.mediapipe.components.CameraHelper.OnCameraStartedListener
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import com.google.protobuf.InvalidProtocolBufferException


class MainActivity : AppCompatActivity() {

    companion object {
        const val LOG_TAG: String = "MainActivity"

        const val BINARY_GRAPH_NAME = "pose_tracking_gpu.binarypb"
        const val ORIGINAL_VIDEO_STREAM = "input_video"
        const val PROCESSED_VIDEO_STREAM = "output_video"
        const val OUTPUT_LANDMARK_STREAM_NAME = "pose_landmarks"

        val CAMERA_FACING_BACK: CameraHelper.CameraFacing = CameraHelper.CameraFacing.BACK
        val CAMERA_FACING_FRONT: CameraHelper.CameraFacing = CameraHelper.CameraFacing.FRONT

        var outputVideoStream: String = PROCESSED_VIDEO_STREAM

        const val FLIP_FRAMES_VERTICALLY = false

        init {
            System.loadLibrary("mediapipe_jni")
            System.loadLibrary("opencv_java3")
        }
    }

    private lateinit var skeletonShowingBtn: Button

    /** Camera Showing in Activity. */
    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private var previewFrameTexture: SurfaceTexture? = null
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private var previewDisplayView: SurfaceView? = null

    // Creates and manages an {@link EGLContext}.
    private var eglManager: EglManager? = null
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private var processor: FrameProcessor? = null
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private var converter: ExternalTextureConverter? = null

    // Handles camera access via the {@link CameraX} Jetpack support library.
    private var cameraHelper: CameraXPreviewHelper? = null


//------------------------------------- LifeCycle Functions ----------------------------------------


    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        Log.i(LOG_TAG, "onCreate, Run.")

        setContentView(R.layout.activity_main)

        skeletonShowingBtn = findViewById(R.id.skeletonShowingBtn)
        skeletonShowingBtn.setOnClickListener {
            toggleSkeletonShowing()
        }

        previewDisplayView = SurfaceView(this)
        setupPreviewDisplayView()

        AndroidAssetUtil.initializeNativeAssetManager(this)
        eglManager = EglManager(null)

        initializeProcessor()

        // Request Camera Permission.
        PermissionHelper.checkAndRequestCameraPermissions(this)

    }

    override fun onResume() {
        super.onResume()
        Log.i(LOG_TAG, "onResume, run.")
        initConverter()
        checkPermissionAndStartCamera()
    }

    override fun onPause() {
        super.onPause()
        Log.i(LOG_TAG, "onPause, run.")
        closeConverter()
    }

    override fun onRestart() {
        super.onRestart()
        Log.i(LOG_TAG, "onRestart, run.")
        initPreviewDisplayView()
    }


//------------------------------------- Processor Functions ---------------------------------------


    private fun initializeProcessor() {
        Log.i(LOG_TAG, "initializeProcessor: run.")
        /**
         * If we change the output stream to null, nothing will show in the screen.
         * If we change the output stream to OUTPUT_VIDEO_STREAM_NAME, skeleton will show.
         * If we change the output stream to INPUT_VIDEO_STREAM_NAME, the camera without skeleton will show.
         * */
        processor = FrameProcessor(
            this,
            eglManager!!.nativeContext,
            BINARY_GRAPH_NAME,
            ORIGINAL_VIDEO_STREAM,
            outputVideoStream
        )

        processor!!.addPacketCallback(
            OUTPUT_LANDMARK_STREAM_NAME
        ) { packet: Packet ->
            try {
                val poseLandmarks = PacketGetter.getProtoBytes(packet)
                val landmarks: NormalizedLandmarkList = NormalizedLandmarkList.parseFrom(poseLandmarks)
                val noseX = landmarks.landmarkList[0].x
                val noseY = landmarks.landmarkList[0].y
                val noseVisibility = landmarks.landmarkList[0].visibility
                Log.i(LOG_TAG, "The position of nose is ( ${noseX}, ${noseY})")
            } catch (exception: InvalidProtocolBufferException) {
                Log.e(LOG_TAG, "Failed to get proto.", exception)
            }
        }
    }



//------------------------------------- Permission Functions ---------------------------------------


    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }


//------------------------------------- Display Functions ------------------------------------------


    private fun initPreviewDisplayView() {
        previewDisplayView = SurfaceView(this)
        setupPreviewDisplayView()
    }

    private fun setupPreviewDisplayView() {

        previewDisplayView?.visibility = View.GONE
        val viewGroup: ViewGroup = findViewById(R.id.preview_display_layout)
        viewGroup.removeAllViews()
        viewGroup.addView(previewDisplayView)

        previewDisplayView?.holder?.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                Log.i(LOG_TAG, "surfaceCreated: run.")
                processor?.videoSurfaceOutput?.setSurface(holder.surface)
            }
            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int
            ) {
                Log.i(LOG_TAG, "surfaceChanged: run.")
                // (Re-)Compute the ideal size of the camera-preview display (the area that the
                // camera-preview frames get rendered onto, potentially with scaling and rotation)
                // based on the size of the SurfaceView that contains the display.
                val viewSize = Size(width, height)
                val displaySize: Size = cameraHelper!!.computeDisplaySizeFromViewSize(viewSize)

                // The camera will rotate the image, so we should rotate back to correct direction.
                // Get the frame size which captures from camera.
                val rotatedDisplaySize: Size = Size(displaySize.height, displaySize.width)

                // Connect the converter to the camera-preview frames as its input (via
                // previewFrameTexture), and configure the output width and height as the computed
                // display size.
                converter!!.setSurfaceTextureAndAttachToGLContext(
                    previewFrameTexture, rotatedDisplaySize.width, rotatedDisplaySize.height
                )
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                Log.i(LOG_TAG, "surfaceDestroyed, run.")
                processor?.videoSurfaceOutput?.setSurface(null)
            }
        })
    }


//------------------------------------- Camera Functions -------------------------------------------


    private fun checkPermissionAndStartCamera() {
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera(CAMERA_FACING_FRONT)
        } else {
            Log.e(LOG_TAG, "Application doesn't have the permission to open camera.")
        }
    }

    private fun startCamera(cameraFacing: CameraHelper.CameraFacing) {
        cameraHelper = CameraXPreviewHelper()
        cameraHelper!!.setOnCameraStartedListener(
            OnCameraStartedListener { surfaceTexture: SurfaceTexture? ->
                previewFrameTexture = surfaceTexture
                // Make the display view visible to start showing the preview.
                previewDisplayView!!.visibility = View.VISIBLE
            })
        cameraHelper
            ?.startCamera(this, cameraFacing, null)
            ?:let { Log.e(LOG_TAG, "Camera Helper is NULL!") }
    }


//------------------------------------- Converter Functions ---------------------------------------


    private fun initConverter() {
        eglManager?.let {
            converter = ExternalTextureConverter(it.context,2)
            converter!!.setFlipY(FLIP_FRAMES_VERTICALLY)
            converter!!.setConsumer(processor)
        }?:let{
            Log.e(LOG_TAG, "EglManager has not be initialized.")
        }
    }

    private fun closeConverter() {
        converter
            ?.close()
            ?:let {
                Log.e(LOG_TAG, "ExternalTextureConverter has not be initialized.")
            }
    }


//------------------------------------- Event Functions --------------------------------------------


    private fun releaseProcessingParam() {
        closeConverter()
        eglManager = null
        converter = null
        processor = null
        previewDisplayView = null
    }

    private fun toggleOutputStream() {
        outputVideoStream = if (outputVideoStream == PROCESSED_VIDEO_STREAM) {
            ORIGINAL_VIDEO_STREAM
        } else {
            PROCESSED_VIDEO_STREAM
        }
    }

    private fun toggleSkeletonShowing() {
        releaseProcessingParam()
        initPreviewDisplayView()
        eglManager = EglManager(null)
        toggleOutputStream()
        initializeProcessor()
        initConverter()
        checkPermissionAndStartCamera()
    }



}