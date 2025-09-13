class CaptionOverlay {
    constructor() {
        this.socket = null;
        this.originalTextElement = document.getElementById('original-text');
        this.translatedTextElement = document.getElementById('translated-text');
        this.currentCaptionTimeout = null;
        this.captionDuration = 5000; // 5 seconds
        this.isConnected = false;
        
        this.initialize();
    }

    initialize() {
        this.connectSocket();
        this.setupEventListeners();
        console.log('Caption overlay initialized');
    }

    connectSocket() {
        try {
            this.socket = io('http://localhost:8080', {
                transports: ['websocket', 'polling'],
                timeout: 5000,
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionAttempts: 5
            });

            this.socket.on('connect', () => {
                console.log('Connected to OBS Live Translator');
                this.isConnected = true;
                this.showStatus('Connected', 'success');
            });

            this.socket.on('disconnect', () => {
                console.log('Disconnected from OBS Live Translator');
                this.isConnected = false;
                this.showStatus('Disconnected', 'error');
            });

            this.socket.on('caption', (data) => {
                this.displayCaption(data);
            });

            this.socket.on('connect_error', (error) => {
                console.error('Connection error:', error);
                this.showStatus('Connection Error', 'error');
            });

        } catch (error) {
            console.error('Failed to initialize socket:', error);
            this.showStatus('Failed to Connect', 'error');
        }
    }

    displayCaption(captionData) {
        if (!captionData) return;

        const { original, translated, timestamp } = captionData;

        // Clear existing timeout
        if (this.currentCaptionTimeout) {
            clearTimeout(this.currentCaptionTimeout);
        }

        // Update original text
        if (original && original.trim()) {
            this.updateTextElement(this.originalTextElement, original);
        }

        // Update translated text
        if (translated && translated.trim()) {
            this.updateTextElement(this.translatedTextElement, translated);
        }

        // Auto-hide captions after duration
        this.currentCaptionTimeout = setTimeout(() => {
            this.hideCaption();
        }, this.captionDuration);

        console.log('Caption displayed:', { original, translated });
    }

    updateTextElement(element, text) {
        // Animate out if there's existing text
        if (element.textContent.trim()) {
            element.classList.add('animate-out');
            
            setTimeout(() => {
                element.textContent = text;
                element.classList.remove('animate-out');
                element.classList.add('show', 'animate-in');
                
                // Remove animation class after animation completes
                setTimeout(() => {
                    element.classList.remove('animate-in');
                }, 500);
            }, 300);
        } else {
            element.textContent = text;
            element.classList.add('show', 'animate-in');
            
            setTimeout(() => {
                element.classList.remove('animate-in');
            }, 500);
        }
    }

    hideCaption() {
        this.originalTextElement.classList.remove('show');
        this.translatedTextElement.classList.remove('show');
        
        // Clear text content after fade out
        setTimeout(() => {
            this.originalTextElement.textContent = '';
            this.translatedTextElement.textContent = '';
        }, 400);
    }

    showStatus(message, type = 'info') {
        // Create temporary status indicator
        const statusElement = document.createElement('div');
        statusElement.className = `status-message status-${type}`;
        statusElement.textContent = message;
        statusElement.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            color: white;
            z-index: 10000;
            opacity: 0.9;
            pointer-events: none;
            background: ${type === 'success' ? '#58CC02' : type === 'error' ? '#FF4444' : '#1CB0F6'};
            animation: slideInRight 0.3s ease-out;
        `;

        document.body.appendChild(statusElement);

        // Remove status after 3 seconds
        setTimeout(() => {
            statusElement.style.animation = 'fadeOut 0.3s ease-out forwards';
            setTimeout(() => {
                if (statusElement.parentNode) {
                    statusElement.parentNode.removeChild(statusElement);
                }
            }, 300);
        }, 3000);
    }

    setupEventListeners() {
        // Handle visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('Overlay hidden');
            } else {
                console.log('Overlay visible');
            }
        });

        // Handle window focus
        window.addEventListener('focus', () => {
            if (!this.isConnected) {
                this.connectSocket();
            }
        });

        // Handle errors
        window.addEventListener('error', (event) => {
            console.error('Overlay error:', event.error);
        });

        // Keyboard shortcuts for testing (only in development)
        if (window.location.hostname === 'localhost') {
            document.addEventListener('keydown', (event) => {
                if (event.key === 't') {
                    // Test caption
                    this.displayCaption({
                        original: 'This is a test caption',
                        translated: '这是一个测试字幕',
                        timestamp: Date.now()
                    });
                } else if (event.key === 'c') {
                    // Clear captions
                    this.hideCaption();
                }
            });
        }
    }

    // Public methods for external control
    setCaptionDuration(duration) {
        this.captionDuration = duration;
    }

    getConnectionStatus() {
        return this.isConnected;
    }

    reconnect() {
        if (this.socket) {
            this.socket.disconnect();
        }
        setTimeout(() => {
            this.connectSocket();
        }, 1000);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 0.9;
            transform: translateX(0);
        }
    }
`;
document.head.appendChild(style);

// Initialize overlay when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.captionOverlay = new CaptionOverlay();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.captionOverlay && window.captionOverlay.socket) {
        window.captionOverlay.socket.disconnect();
    }
});