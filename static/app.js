document.addEventListener(
    "DOMContentLoaded",
    function () {

        "use strict";


        // ====================================================
        // ELEMENTS
        // ====================================================

        const model =
            document.getElementById("model");

        const confidence =
            document.getElementById("confidence");

        const confidenceValue =
            document.getElementById(
                "confidenceValue"
            );

        const iou =
            document.getElementById("iou");

        const iouValue =
            document.getElementById(
                "iouValue"
            );


        const imageInput =
            document.getElementById(
                "imageInput"
            );

        const videoInput =
            document.getElementById(
                "videoInput"
            );


        const selectedImageName =
            document.getElementById(
                "selectedImageName"
            );

        const selectedVideoName =
            document.getElementById(
                "selectedVideoName"
            );


        const startWebcam =
            document.getElementById(
                "startWebcam"
            );

        const stopWebcam =
            document.getElementById(
                "stopWebcam"
            );


        const camera =
            document.getElementById(
                "camera"
            );

        const cameraCanvas =
            document.getElementById(
                "cameraCanvas"
            );


        const resultImage =
            document.getElementById(
                "resultImage"
            );

        const resultVideo =
            document.getElementById(
                "resultVideo"
            );

        const emptyState =
            document.getElementById(
                "emptyState"
            );


        const status =
            document.getElementById(
                "status"
            );


        const total =
            document.getElementById(
                "total"
            );

        const cylinder =
            document.getElementById(
                "cylinder"
            );

        const shock =
            document.getElementById(
                "shock"
            );

        const avgConfidence =
            document.getElementById(
                "avgConfidence"
            );

        const latency =
            document.getElementById(
                "latency"
            );


        const download =
            document.getElementById(
                "download"
            );


        console.log(
            "HazWaste Detection JavaScript loaded."
        );


        // ====================================================
        // STATUS
        // ====================================================

        function setStatus(message) {

            if (status) {

                status.textContent =
                    message;

            }

        }


        // ====================================================
        // SLIDERS
        // ====================================================

        function updateSliders() {

            if (confidence && confidenceValue) {

                confidenceValue.textContent =
                    Number(
                        confidence.value
                    ).toFixed(2);

            }


            if (iou && iouValue) {

                iouValue.textContent =
                    Number(
                        iou.value
                    ).toFixed(2);

            }

        }


        if (confidence) {

            confidence.addEventListener(
                "input",
                updateSliders
            );

        }


        if (iou) {

            iou.addEventListener(
                "input",
                updateSliders
            );

        }


        updateSliders();


        // ====================================================
        // MODEL CHANGE
        // ====================================================

        if (model) {

            model.addEventListener(
                "change",
                function () {

                    console.log(
                        "Selected model:",
                        model.value
                    );

                    setStatus(
                        model.value +
                        " selected"
                    );

                }
            );

        }


        // ====================================================
        // TABS
        // ====================================================

        const tabs =
            document.querySelectorAll(
                ".tab"
            );

        const panels =
            document.querySelectorAll(
                ".panel"
            );


        let webcamStream =
            null;

        let webcamTimer =
            null;

        let webcamBusy =
            false;


        tabs.forEach(
            function (tab) {

                tab.addEventListener(
                    "click",
                    function () {

                        tabs.forEach(
                            function (item) {

                                item.classList.remove(
                                    "active"
                                );

                            }
                        );


                        panels.forEach(
                            function (panel) {

                                panel.classList.remove(
                                    "active"
                                );

                            }
                        );


                        tab.classList.add(
                            "active"
                        );


                        const target =
                            document.getElementById(
                                tab.dataset.panel
                            );


                        if (target) {

                            target.classList.add(
                                "active"
                            );

                        }


                        // Stop webcam when
                        // leaving webcam tab

                        if (
                            tab.dataset.panel !==
                            "webcamPanel"
                        ) {

                            if (webcamStream) {

                                stopCamera();

                            }

                        }

                    }
                );

            }
        );


        // ====================================================
        // IMAGE FILE SELECTION
        // ====================================================

        if (imageInput) {

            imageInput.addEventListener(
                "change",
                function () {

                    if (
                        !imageInput.files ||
                        imageInput.files.length === 0
                    ) {

                        return;

                    }


                    const file =
                        imageInput.files[0];


                    if (selectedImageName) {

                        selectedImageName.textContent =
                            file.name;

                    }


                    processImage(
                        file
                    );

                }
            );

        }


        // ====================================================
        // IMAGE DETECTION
        // ====================================================

        async function processImage(
            file
        ) {

            hideResults();


            setStatus(
                "Uploading image..."
            );


            const formData =
                new FormData();


            // IMPORTANT:
            // FastAPI expects "file"

            formData.append(
                "file",
                file
            );


            // Selected YOLO model

            formData.append(
                "model",
                model.value
            );


            // Confidence

            formData.append(
                "confidence",
                confidence.value
            );


            // IoU

            formData.append(
                "iou",
                iou.value
            );


            try {

                console.log(
                    "Sending image..."
                );

                console.log(
                    "Model:",
                    model.value
                );


                const response =
                    await fetch(
                        "/api/image",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        data.error ||
                        "Image detection failed."
                    );

                }


                // =================================================
                // DISPLAY RESULT IMAGE
                // =================================================

                if (data.image) {

                    resultImage.src =
                        data.image +
                        "&t=" +
                        Date.now();

                }
                else if (data.url) {

                    resultImage.src =
                        data.url +
                        "?t=" +
                        Date.now();

                }


                resultImage.hidden =
                    false;


                resultVideo.hidden =
                    true;


                emptyState.hidden =
                    true;


                // =================================================
                // DOWNLOAD
                // =================================================

                if (data.url) {

                    download.href =
                        data.url;

                    download.download =
                        "hazwaste_detection_result.jpg";

                    download.hidden =
                        false;

                }


                // =================================================
                // UPDATE STATISTICS
                // =================================================

                updateStats(
                    data
                );


                // =================================================
                // STATUS
                // =================================================

                setStatus(
                    (data.stats &&
                        data.stats.model
                    )
                    ||
                    data.model
                    ||
                    model.value
                    + " • Image complete"
                );


                console.log(
                    "Image detection complete:",
                    data
                );

            }
            catch (error) {

                console.error(
                    "Image error:",
                    error
                );


                showError(
                    error.message
                );

            }

        }


        // ====================================================
        // VIDEO FILE SELECTION
        // ====================================================

        if (videoInput) {

            videoInput.addEventListener(
                "change",
                function () {

                    if (
                        !videoInput.files ||
                        videoInput.files.length === 0
                    ) {

                        return;

                    }


                    const file =
                        videoInput.files[0];


                    if (selectedVideoName) {

                        selectedVideoName.textContent =
                            file.name;

                    }


                    processVideo(
                        file
                    );

                }
            );

        }


        // ====================================================
        // VIDEO DETECTION
        // ====================================================

        async function processVideo(
            file
        ) {

            hideResults();


            setStatus(
                "Uploading and processing video..."
            );


            const formData =
                new FormData();


            // IMPORTANT:
            // FastAPI expects "file"

            formData.append(
                "file",
                file
            );


            formData.append(
                "model",
                model.value
            );


            formData.append(
                "confidence",
                confidence.value
            );


            formData.append(
                "iou",
                iou.value
            );


            try {

                console.log(
                    "Sending video..."
                );

                console.log(
                    "Model:",
                    model.value
                );


                const response =
                    await fetch(
                        "/api/video",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        data.error ||
                        "Video detection failed."
                    );

                }


                // =================================================
                // DISPLAY RESULT VIDEO
                // =================================================

                resultImage.hidden =
                    true;


                resultVideo.hidden =
                    false;


                emptyState.hidden =
                    true;


                if (data.url) {

                    resultVideo.src =
                        data.url +
                        "?t=" +
                        Date.now();

                    resultVideo.load();

                }


                // =================================================
                // DOWNLOAD
                // =================================================

                if (data.url) {

                    download.href =
                        data.url;

                    download.download =
                        "hazwaste_detection_result.mp4";

                    download.hidden =
                        false;

                }


                // =================================================
                // UPDATE STATISTICS
                // =================================================

                updateStats(
                    data
                );


                // =================================================
                // STATUS
                // =================================================

                const stats =
                    data.stats || {};


                const frames =
                    stats.frames || 0;


                setStatus(
                    (
                        stats.model ||
                        data.model ||
                        model.value
                    )
                    +
                    " • "
                    +
                    frames
                    +
                    " frames processed"
                );


                console.log(
                    "Video detection complete:",
                    data
                );

            }
            catch (error) {

                console.error(
                    "Video error:",
                    error
                );


                showError(
                    error.message
                );

            }

        }


        // ====================================================
        // WEBCAM
        // ====================================================

        if (startWebcam) {

            startWebcam.addEventListener(
                "click",
                startCamera
            );

        }


        if (stopWebcam) {

            stopWebcam.addEventListener(
                "click",
                stopCamera
            );

        }


        // ====================================================
        // START CAMERA
        // ====================================================

        async function startCamera() {

            if (webcamStream) {

                return;

            }


            try {

                if (
                    !navigator.mediaDevices ||
                    !navigator.mediaDevices.getUserMedia
                ) {

                    throw new Error(
                        "Webcam is not supported by this browser."
                    );

                }


                setStatus(
                    "Requesting camera permission..."
                );


                webcamStream =
                    await navigator
                        .mediaDevices
                        .getUserMedia(
                            {
                                video: {
                                    width: {
                                        ideal: 1280
                                    },
                                    height: {
                                        ideal: 720
                                    }
                                },

                                audio: false
                            }
                        );


                camera.srcObject =
                    webcamStream;


                await camera.play();


                startWebcam.disabled =
                    true;


                stopWebcam.disabled =
                    false;


                setStatus(
                    "Webcam connected."
                );


                // Send one frame every 600 ms

                webcamTimer =
                    setInterval(
                        sendWebcamFrame,
                        600
                    );

            }
            catch (error) {

                console.error(
                    "Webcam error:",
                    error
                );


                stopCamera();


                showError(
                    "Could not access webcam.\n\n"
                    +
                    error.message
                );

            }

        }


        // ====================================================
        // STOP CAMERA
        // ====================================================

        function stopCamera() {

            if (
                webcamTimer !==
                null
            ) {

                clearInterval(
                    webcamTimer
                );


                webcamTimer =
                    null;

            }


            if (webcamStream) {

                webcamStream
                    .getTracks()
                    .forEach(
                        function (track) {

                            track.stop();

                        }
                    );


                webcamStream =
                    null;

            }


            if (camera) {

                camera.srcObject =
                    null;

            }


            webcamBusy =
                false;


            if (startWebcam) {

                startWebcam.disabled =
                    false;

            }


            if (stopWebcam) {

                stopWebcam.disabled =
                    true;

            }


            setStatus(
                "Webcam stopped."
            );

        }


        // ====================================================
        // SEND WEBCAM FRAME
        // ====================================================

        async function sendWebcamFrame() {

            if (
                webcamBusy ||
                !webcamStream ||
                !camera.videoWidth ||
                !camera.videoHeight
            ) {

                return;

            }


            webcamBusy =
                true;


            try {

                // =================================================
                // SET CANVAS SIZE
                // =================================================

                cameraCanvas.width =
                    camera.videoWidth;


                cameraCanvas.height =
                    camera.videoHeight;


                const context =
                    cameraCanvas.getContext(
                        "2d"
                    );


                // =================================================
                // COPY CAMERA FRAME
                // =================================================

                context.drawImage(
                    camera,
                    0,
                    0,
                    cameraCanvas.width,
                    cameraCanvas.height
                );


                // =================================================
                // CONVERT FRAME TO JPEG
                // =================================================

                const blob =
                    await new Promise(
                        function (resolve) {

                            cameraCanvas.toBlob(
                                resolve,
                                "image/jpeg",
                                0.78
                            );

                        }
                    );


                if (!blob) {

                    throw new Error(
                        "Could not capture webcam frame."
                    );

                }


                // =================================================
                // READ IMAGE AS BASE64
                // =================================================

                const reader =
                    new FileReader();


                const base64Image =
                    await new Promise(
                        function (resolve, reject) {

                            reader.onload =
                                function () {

                                    resolve(
                                        reader.result
                                    );

                                };


                            reader.onerror =
                                function () {

                                    reject(
                                        new Error(
                                            "Could not convert webcam frame."
                                        )
                                    );

                                };


                            reader.readAsDataURL(
                                blob
                            );

                        }
                    );


                // =================================================
                // FORM DATA
                // =================================================

                const formData =
                    new FormData();


                // IMPORTANT:
                // FastAPI expects "image"

                formData.append(
                    "image",
                    base64Image
                );


                formData.append(
                    "model",
                    model.value
                );


                formData.append(
                    "confidence",
                    confidence.value
                );


                formData.append(
                    "iou",
                    iou.value
                );


                // =================================================
                // SEND TO FASTAPI
                // =================================================

                const response =
                    await fetch(
                        "/api/frame",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        data.error ||
                        "Webcam detection failed."
                    );

                }


                // =================================================
                // DISPLAY RESULT
                // =================================================

                if (data.image) {

                    // Backend already returns:
                    // data:image/jpeg;base64,...

                    if (
                        data.image.startsWith(
                            "data:"
                        )
                    ) {

                        resultImage.src =
                            data.image;

                    }
                    else {

                        resultImage.src =
                            "data:image/jpeg;base64,"
                            +
                            data.image;

                    }

                }


                resultImage.hidden =
                    false;


                resultVideo.hidden =
                    true;


                emptyState.hidden =
                    true;


                // =================================================
                // UPDATE STATISTICS
                // =================================================

                updateStats(
                    data
                );


                // =================================================
                // STATUS
                // =================================================

                setStatus(
                    (
                        data.stats &&
                        data.stats.model
                    )
                    ||
                    data.model
                    ||
                    model.value
                )
                ;


            }
            catch (error) {

                console.error(
                    "Webcam frame error:",
                    error
                );


                setStatus(
                    "Webcam detection error."
                );

            }
            finally {

                webcamBusy =
                    false;

            }

        }


        // ====================================================
        // UPDATE STATISTICS
        // ====================================================

        function updateStats(
            data
        ) {

            /*
             * FastAPI response format:
             *
             * {
             *     "success": true,
             *     "stats": {
             *         "model": "...",
             *         "detections": 2,
             *         "class_counts": {
             *             "cylinder": 1,
             *             "shock_absorber": 1
             *         },
             *         "inference_ms": 25.5
             *     }
             * }
             */


            const stats =
                data.stats || data;


            // =================================================
            // TOTAL OBJECTS
            // =================================================

            const totalObjects =
                stats.detections ||
                stats.total ||
                0;


            total.textContent =
                totalObjects;


            // =================================================
            // CLASS COUNTS
            // =================================================

            const classCounts =
                stats.class_counts ||
                stats.counts ||
                {};


            // -------------------------------------------------
            // Cylinder
            // -------------------------------------------------

            let cylinderCount =
                0;


            // Handle different possible
            // capitalization formats

            Object.keys(
                classCounts
            ).forEach(
                function (className) {

                    const normalized =
                        className
                            .toLowerCase()
                            .replace(
                                /[\s-]+/g,
                                "_"
                            );


                    if (
                        normalized ===
                        "cylinder"
                    ) {

                        cylinderCount +=
                            Number(
                                classCounts[
                                    className
                                ]
                            ) || 0;

                    }

                }
            );


            cylinder.textContent =
                cylinderCount;


            // -------------------------------------------------
            // Shock absorber
            // -------------------------------------------------

            let shockCount =
                0;


            Object.keys(
                classCounts
            ).forEach(
                function (className) {

                    const normalized =
                        className
                            .toLowerCase()
                            .replace(
                                /[\s-]+/g,
                                "_"
                            );


                    if (
                        normalized ===
                            "shock_absorber"
                        ||
                        normalized ===
                            "shockabsorber"
                    ) {

                        shockCount +=
                            Number(
                                classCounts[
                                    className
                                ]
                            ) || 0;

                    }

                }
            );


            shock.textContent =
                shockCount;


            // =================================================
            // AVERAGE CONFIDENCE
            // =================================================

            /*
             * The current FastAPI backend does not
             * calculate average confidence.
             *
             * Therefore we display 0.00 unless
             * the backend supplies it.
             */

            const average =
                stats.average_confidence;


            if (
                average !== undefined &&
                average !== null
            ) {

                avgConfidence.textContent =
                    Number(
                        average
                    ).toFixed(2);

            }
            else {

                avgConfidence.textContent =
                    "0.00";

            }


            // =================================================
            // INFERENCE TIME
            // =================================================

            const inference =
                stats.inference_ms ||
                stats.latency_ms ||
                0;


            latency.textContent =
                Number(
                    inference
                ).toFixed(1)
                +
                " ms";

        }


        // ====================================================
        // HIDE RESULTS
        // ====================================================

        function hideResults() {

            resultImage.hidden =
                true;


            resultVideo.hidden =
                true;


            emptyState.hidden =
                false;


            download.hidden =
                true;


            // Reset statistics

            total.textContent =
                "0";


            cylinder.textContent =
                "0";


            shock.textContent =
                "0";


            avgConfidence.textContent =
                "0.00";


            latency.textContent =
                "0 ms";

        }


        // ====================================================
        // SHOW ERROR
        // ====================================================

        function showError(
            message
        ) {

            setStatus(
                "Error"
            );


            window.alert(
                message
            );

        }


        // ====================================================
        // CLEAN SHUTDOWN
        // ====================================================

        window.addEventListener(
            "beforeunload",
            function () {

                stopCamera();

            }
        );


        // ====================================================
        // INITIAL STATE
        // ====================================================

        setStatus(
            "Ready"
        );


        console.log(
            "HazWaste Detection UI initialized successfully."
        );

    }
);