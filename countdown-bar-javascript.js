// Add these JavaScript functions to your existing <script> section in index.html

let countdownInterval = null;

// Modify your existing performSafetyCheck() to call handleCountdown
// Add this after updateSafetyUI(data) line:
// handleCountdown(data);

function handleCountdown(data) {
    const bar = document.getElementById('countdownBar');
    const timer = document.getElementById('countdownBarTimer');
    
    if (data.critical_countdown_active && data.critical_countdown_remaining !== null) {
        bar.classList.add('active');
        timer.textContent = data.critical_countdown_remaining;
        
        if (countdownInterval) clearInterval(countdownInterval);
        
        countdownInterval = setInterval(() => {
            performSafetyCheck();
        }, 1000);
        
        if (data.critical_countdown_remaining === 0) {
            clearInterval(countdownInterval);
            initiateEmergencyShutdown();
        }
    } else {
        bar.classList.remove('active');
        if (countdownInterval) {
            clearInterval(countdownInterval);
            countdownInterval = null;
        }
    }
}

async function acknowledgeWarning() {
    try {
        await fetch(`${API_BASE}/acknowledge_critical`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ acknowledged: true })
        });
        document.getElementById('countdownBar').classList.remove('active');
        if (countdownInterval) {
            clearInterval(countdownInterval);
            countdownInterval = null;
        }
    } catch (error) {
        console.error('Failed to acknowledge:', error);
    }
}

function shutdownFromCountdown() {
    document.getElementById('countdownBar').classList.remove('active');
    if (countdownInterval) {
        clearInterval(countdownInterval);
        countdownInterval = null;
    }
    initiateEmergencyShutdown();
}