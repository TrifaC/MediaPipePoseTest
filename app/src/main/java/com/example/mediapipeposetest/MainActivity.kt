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
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import com.google.mediapipe.components.*
import com.google.mediapipe.components.CameraHelper.OnCameraStartedListener
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.glutil.EglManager


class MainActivity : AppCompatActivity() {

    companion object {
        const val LOG_TAG: String = "MainActivity"

        const val BINARY_GRAPH_NAME = "pose_tracking_gpu.binarypb"
        const val INPUT_VIDEO_STREAM_NAME = "input_video"
        const val OUTPUT_VIDEO_STREAM_NAME = "output_video"
        const val OUTPUT_LANDMARK_STREAM_NAME = "pose_landmarks"
        const val NUM_HANDS = 2
        val CAMERA_FACING: CameraHelper.CameraFacing = CameraHelper.CameraFacing.FRONT
        const val FLIP_FRAMES_VERTICALLY = false

        init {
            System.loadLibrary("mediapipe_jni")
            System.loadLibrary("opencv_java3")
        }
    }

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
        setContentView(R.layout.activity_main)

        previewDisplayView = SurfaceView(this)
        setupPreviewDisplayView()

        AndroidAssetUtil.initializeNativeAssetManager(this)
        eglManager = EglManager(null)

        processor = FrameProcessor(
            this,
            eglManager!!.nativeContext,
            BINARY_GRAPH_NAME,
            INPUT_VIDEO_STREAM_NAME,
            OUTPUT_VIDEO_STREAM_NAME
        )

        // Request Camera Permission.
        PermissionHelper.checkAndRequestCameraPermissions(this)
    }

    override fun onResume() {
        super.onResume()
        eglManager?.let {
            converter = ExternalTextureConverter(it.context,2)
            converter!!.setFlipY(FLIP_FRAMES_VERTICALLY)
            converter!!.setConsumer(processor)
        }?:let{
            Log.e(LOG_TAG, "EglManager has not be initialized.")
        }
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera()
        }
    }

    override fun onPause() {
        super.onPause()
        converter?.close() ?:let { Log.e(LOG_TAG, "ExternalTextureConverter has not be initialized.") }
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


    private fun setupPreviewDisplayView() {

        previewDisplayView?.visibility = View.GONE
        val viewGroup: ViewGroup = findViewById(R.id.preview_display_layout)
        viewGroup.addView(previewDisplayView)

        previewDisplayView?.holder?.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                processor?.videoSurfaceOutput?.setSurface(holder.surface)
            }
            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int
            ) {
                // (Re-)Compute the ideal size of the camera-preview display (the area that the
                // camera-preview frames get rendered onto, potentially with scaling and rotation)
                // based on the size of the SurfaceView that contains the display.
                val viewSize = Size(width, height)
                val displaySize: Size = cameraHelper!!.computeDisplaySizeFromViewSize(viewSize)
                // The camera will rotate the image, so we should rotate back to correct direction.
                val rotatedDisplaySize: Size = Size(displaySize.height, displaySize.width)

                // Connect the converter to the camera-preview frames as its input (via
                // previewFrameTexture), and configure the output width and height as the computed
                // display size.
                converter!!.setSurfaceTextureAndAttachToGLContext(
                    previewFrameTexture, rotatedDisplaySize.width, rotatedDisplaySize.height
                )
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                processor?.videoSurfaceOutput?.setSurface(null)
            }
        })
    }


//------------------------------------- Camera Functions -------------------------------------------

    fun startCamera() {
        cameraHelper = CameraXPreviewHelper()
        cameraHelper!!.setOnCameraStartedListener(
            OnCameraStartedListener { surfaceTexture: SurfaceTexture? ->
                previewFrameTexture = surfaceTexture
                // Make the display view visible to start showing the preview.
                previewDisplayView!!.visibility = View.VISIBLE
            })
        cameraHelper
            ?.startCamera(this, CAMERA_FACING, null)
            ?:let { Log.e(LOG_TAG, "Camera Helper is NULL!") }
    }
}