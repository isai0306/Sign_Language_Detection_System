/**
 * SignAI - Real-time Gesture Detection
 * Handles camera, frame capture, and API communication
 */

class GestureDetector {
    constructor(config = {}) {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        
        // Configuration
        this.detectionInterval = config.detectionInterval || 500; // ms
        this.isDetecting = false;
        this.detectionTimer = null;
        this.currentGesture = null;
        this.gestureStableCount = 0;
        this.requiredStableFrames = config.requiredStableFrames || 3;
        
        // Callbacks
        this.onGestureDetected = config.onGestureDetected || (() => {});
        this.onEmergency = config.onEmergency || (() => {});
        this.onError = config.onError || ((err) => console.error(err));
        
        // Audio elements
        this.audioElements = {};
    }
    
    /**
     * Initialize camera and canvas
     */
    async init(videoElementId, canvasElementId) {
        try {
            this.video = document.getElementById(videoElementId);
            this.canvas = document.getElementById(canvasElementId);
            
            if (!this.video || !this.canvas) {
                throw new Error('Video or canvas element not found');
            }
            
            this.ctx = this.canvas.getContext('2d');
            
            // Get camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            this.video.srcObject = this.stream;
            
            // Wait for video to load
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });
            
            // Set canvas size to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            
            console.log('✅ Camera initialized');
            return true;
            
        } catch (error) {
            this.onError(`Camera initialization failed: ${error.message}`);
            return false;
        }
    }
    
    /**
     * Start gesture detection
     */
    startDetection() {
        if (this.isDetecting) {
            console.log('Detection already running');
            return;
        }
        
        this.isDetecting = true;
        console.log('🎥 Starting gesture detection...');
        
        this.detectionTimer = setInterval(() => {
            this.detectGesture();
        }, this.detectionInterval);
    }
    
    /**
     * Stop gesture detection
     */
    stopDetection() {
        this.isDetecting = false;
        
        if (this.detectionTimer) {
            clearInterval(this.detectionTimer);
            this.detectionTimer = null;
        }
        
        console.log('⏸️ Detection stopped');
    }
    
    /**
     * Capture current frame and detect gesture
     */
    async detectGesture() {
        if (!this.video || !this.canvas || !this.isDetecting) {
            return;
        }
        
        try {
            // Draw current video frame to canvas
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert canvas to base64
            const frameData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to backend for detection
            const response = await fetch('/api/detect-gesture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    frame: frameData,
                    user_id: this.getUserId()
                })
            });
            
            const result = await response.json();
            
            if (result.success && result.gesture) {
                this.handleGestureResult(result);
            }
            
        } catch (error) {
            console.error('Detection error:', error);
        }
    }
    
    /**
     * Handle gesture detection result
     */
    handleGestureResult(result) {
        const { gesture, confidence, translations, audio_paths, is_emergency } = result;
        
        // Check if gesture is stable (same gesture detected multiple times)
        if (gesture === this.currentGesture) {
            this.gestureStableCount++;
        } else {
            this.currentGesture = gesture;
            this.gestureStableCount = 1;
        }
        
        // Only trigger callback if gesture is stable
        if (this.gestureStableCount >= this.requiredStableFrames) {
            // Trigger gesture detected callback
            this.onGestureDetected({
                gesture,
                confidence,
                translations,
                timestamp: new Date()
            });
            
            // Play audio if available
            if (audio_paths) {
                this.playAudio(audio_paths);
            }
            
            // Handle emergency
            if (is_emergency) {
                this.handleEmergency(gesture, translations);
            }
            
            // Reset counter
            this.gestureStableCount = 0;
            this.currentGesture = null;
        }
    }
    
    /**
     * Play audio for detected gesture
     */
    playAudio(audio_paths) {
        try {
            // Play all available audio files
            Object.entries(audio_paths).forEach(([lang, path]) => {
                if (path) {
                    const audio = new Audio(`/static/${path}`);
                    audio.play().catch(err => console.log('Audio play failed:', err));
                }
            });
        } catch (error) {
            console.error('Audio playback error:', error);
        }
    }
    
    /**
     * Handle emergency gesture
     */
    async handleEmergency(gesture, translations) {
        console.warn('🚨 EMERGENCY DETECTED:', gesture);
        
        // Trigger emergency callback
        this.onEmergency({
            gesture,
            translations,
            timestamp: new Date()
        });
        
        // Send emergency alert to backend
        try {
            await fetch('/api/emergency-alert', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: this.getUserId(),
                    gesture_name: gesture,
                    location_info: this.getLocationInfo()
                })
            });
        } catch (error) {
            console.error('Emergency alert failed:', error);
        }
    }
    
    /**
     * Get user ID from session or page
     */
    getUserId() {
        // Try to get from data attribute or hidden input
        const userIdElement = document.querySelector('[data-user-id]');
        if (userIdElement) {
            return userIdElement.getAttribute('data-user-id');
        }
        return null;
    }
    
    /**
     * Get location information (placeholder)
     */
    getLocationInfo() {
        // In production, you could use Geolocation API
        return 'Location unavailable';
    }
    
    /**
     * Take snapshot
     */
    takeSnapshot() {
        if (!this.canvas) return null;
        return this.canvas.toDataURL('image/png');
    }
    
    /**
     * Stop camera and cleanup
     */
    stop() {
        this.stopDetection();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.video) {
            this.video.srcObject = null;
        }
        
        console.log('🛑 Camera stopped');
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GestureDetector;
}