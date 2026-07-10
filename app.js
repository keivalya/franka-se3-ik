// app.js - Simplified JavaScript for Academic Paper-style Companion Website

// Metric data for each alpha value
const alphaMetrics = {
    "alpha_0.00": {
        title: "Passive Nullspace (alpha = 0.00)",
        desc: "Joint Centering disabled. Redundancy is unresolved. Wrist joints 4 and 6 wander near physical limits, risking joint locking and mechanical stress.",
        metric: "Mean Dev: 39.5° | Peak Pos Error: 0.1 μm | Joint centering: OFF"
    },
    "alpha_0.10": {
        title: "Mild Centering (alpha = 0.10)",
        desc: "Stabilizes joints, pulling them away from boundaries. Reconciles sub-micron Cartesian tracking accuracy with a 5.4% mean joint deviation reduction.",
        metric: "Mean Dev: 37.3° | Peak Pos Error: 0.6 μm | Joint centering: STABLE"
    },
    "alpha_0.30": {
        title: "Moderate Centering (alpha = 0.30)",
        desc: "Stronger centering force. Limits maximum joint ranges, reducing joint deviation from midpoint by 10.9%. Cartesian tracking remains accurate to the micron level.",
        metric: "Mean Dev: 35.2° | Peak Pos Error: 3.1 μm | Joint centering: ACTIVE"
    },
    "alpha_0.50": {
        title: "Active Centering (alpha = 0.50)",
        desc: "Maximum corrective force. Achieves a 13.3% overall joint range reduction. A transient Cartesian drift (up to 76.9 mm) occurs as the null-space velocity vectors accumulate over finite step size integration.",
        metric: "Mean Dev: 34.2° | Peak Pos Error: 76.9 mm | Joint centering: MAX"
    }
};

window.selectAlpha = function(alphaVal) {
    const key = `alpha_${alphaVal.toFixed(2)}`;
    
    // 1. Highlight active button
    document.querySelectorAll('.alpha-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.includes(`Alpha = ${alphaVal.toFixed(2)}`)) {
            btn.classList.add('active');
        }
    });
    
    // 2. Update Video Player
    const video = document.getElementById('alpha-video');
    if (video) {
        video.src = `assets/videos/${key}.mp4`;
        video.load();
        video.play().catch(e => {
            // Silence autoplay restrictions
            console.log("Autoplay delayed until user interaction.");
        });
    }
    
    // 3. Update Text Descriptions
    const titleEl = document.getElementById('nullspace-impact-title');
    const descEl = document.getElementById('nullspace-impact-desc');
    const metricEl = document.getElementById('nullspace-widget-metric');
    
    if (titleEl && descEl && metricEl) {
        titleEl.textContent = alphaMetrics[key].title;
        descEl.textContent = alphaMetrics[key].desc;
        metricEl.textContent = alphaMetrics[key].metric;
    }
};

// Initialize default view to alpha = 0.50
window.addEventListener('DOMContentLoaded', () => {
    selectAlpha(0.50);
});
