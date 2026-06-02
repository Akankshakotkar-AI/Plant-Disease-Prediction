document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const browseBtn = document.querySelector('.browse-btn');
    const dropContent = document.querySelector('.drop-content');
    
    const predictBtn = document.getElementById('predict-btn');
    const cameraBtn = document.getElementById('camera-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const cameraModal = document.getElementById('camera-modal');
    const video = document.getElementById('video');
    const captureBtn = document.getElementById('capture-btn');
    const closeCameraBtn = document.getElementById('close-camera');
    
    const resultSection = document.getElementById('result-section');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const resultContent = document.getElementById('result-content');
    
    const plantNameDisplay = document.getElementById('plant-name');
    const statusBadge = document.getElementById('status-badge');
    const diseaseNameDisplay = document.getElementById('disease-name');
    const confidencePct = document.getElementById('confidence-pct');
    const confidenceFill = document.getElementById('confidence-fill');
    
    const navStatus = document.getElementById('system-status');
    const statusDot = navStatus.querySelector('.status-dot');
    const statusText = document.getElementById('status-text');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // --- Configuration ---
    const API_BASE = 'http://localhost:5000';

    // --- State ---
    let selectedFile = null;
    let stream = null;
    let isAIReady = false;

    // --- Initialization ---
    initializeAIControl();

    async function initializeAIControl() {
        showError(null);
        statusDot.className = 'status-dot';
        statusText.textContent = 'Contacting AI Backend...';
        predictBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/health`, { 
                mode: 'cors'
            });
            const data = await response.json();
            
            if (data.status === 'online') {
                if (data.ai_engine === 'READY') {
                    isAIReady = true;
                    statusDot.className = 'status-dot online';
                    statusText.textContent = 'AI System: Active';
                    if (selectedFile) predictBtn.disabled = false;
                } else {
                    isAIReady = false;
                    statusDot.className = 'status-dot warning';
                    statusText.textContent = 'AI Status: Model Missing (.h5 Not Found)';
                    showError("Real AI inference is currently disabled. Please generate and place 'plant_disease_model.h5' in the backend folder.");
                }
            }
        } catch (err) {
            isAIReady = false;
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'AI Status: Server Offline';
            showError("Cannot connect to AI engine. Ensure 'app.py' is running on port 5000.");
        }
    }

    function showError(msg) {
        if (msg) {
            errorMessage.textContent = msg;
            errorContainer.classList.remove('hidden');
        } else {
            errorContainer.classList.add('hidden');
        }
    }

    // --- Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', handleFileSelect);

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('active');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    }

    function handleFile(file) {
        showError(null);
        if (!file.type.startsWith('image/')) {
            showError('Invalid file type. Please upload a plant leaf image (JPG/PNG).');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropContent.classList.add('hidden');
            if (isAIReady) predictBtn.disabled = false;
            resetBtn.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // --- Camera Logic ---
    cameraBtn.addEventListener('click', async () => {
        showError(null);
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment' } 
            });
            video.srcObject = stream;
            cameraModal.classList.remove('hidden');
        } catch (err) {
            showError("Camera access denied. Please check permissions.");
        }
    });

    captureBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0);
        
        canvas.toBlob((blob) => {
            const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
            handleFile(file);
            stopCamera();
        }, 'image/jpeg');
    });

    closeCameraBtn.addEventListener('click', stopCamera);

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        cameraModal.classList.add('hidden');
    }

    // --- Real Prediction Logic ---
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile || !isAIReady) return;
        showError(null);

        // UI Prep
        resultSection.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');
        resultContent.classList.add('hidden');
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.error && data.error !== 'Low confidence prediction') {
                throw new Error(data.message || data.error);
            }

            displayResult(data);

        } catch (err) {
            console.error("AI engine failure:", err);
            showError(`Engine Error: ${err.message}. Check if backend is running at ${API_BASE}`);
            resultSection.classList.add('hidden');
        } finally {
            loadingOverlay.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    function displayResult(data) {
        resultContent.classList.remove('hidden');

        plantNameDisplay.textContent = data.plant;
        diseaseNameDisplay.textContent = data.disease;
        const status = data.status || 'Unknown';
        statusBadge.textContent = status;
        
        // Badge color logic
        if (status.toLowerCase() === 'healthy') {
            statusBadge.className = 'badge';
        } else {
            statusBadge.className = 'badge diseased';
        }

        confidencePct.textContent = `${data.confidence}%`;
        confidenceFill.style.width = `${data.confidence}%`;

        // Update Report Content with professional agricultural data
        const report = data.report || {};
        document.getElementById('disease-desc').textContent = report.description || 'Information pending for this variety.';
        document.getElementById('disease-symptoms').textContent = report.symptoms || 'Visual indicators vary; consult images.';
        document.getElementById('disease-treatment').textContent = report.treatment || 'Check local agricultural advisories.';
        document.getElementById('disease-causes').textContent = report.causes || 'Pathogen or environmental factors.';
        document.getElementById('disease-prevention').textContent = report.prevention || 'Maintain crop hygiene and climate control.';

        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    // --- Tab Switching Logic ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
        });
    });

    // --- UI Reset ---
    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.classList.add('hidden');
        imagePreview.src = '';
        dropContent.classList.remove('hidden');
        predictBtn.disabled = true;
        resetBtn.classList.add('hidden');
        resultSection.classList.add('hidden');
        showError(null);
    });
});
