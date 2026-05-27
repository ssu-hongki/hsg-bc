const GLOVE_BOXES = [
    { id: "box-1", name: "Glove Box #1", status: "available", desc: "General Synthesis" },
    { id: "box-2", name: "Glove Box #2", status: "offline", desc: "Cell Culture Prep" },
    { id: "box-3", name: "Glove Box #3", status: "offline", desc: "Material Analysis" },
    { id: "box-4", name: "Glove Box #4", status: "offline", desc: "Maintenance Required" },
];

const TEXT = {
    greeting: "\uC791\uC5C5 \uBA85\uB839 \uB300\uAE30\uC911...",
    autoPrompt: "\uC0D8\uD50C \uC900\uBE44, \uC774\uC1A1, \uBC18\uC751, \uC815\uB9AC\uAE4C\uC9C0 \uC804\uCCB4 \uACF5\uC815\uC744 \uC790\uB3D9\uC73C\uB85C \uC218\uD589\uD574\uC918.",
    standby: "\uB300\uAE30 \uC911",
    planning: "LLM\uC774 \uC791\uC5C5 \uC808\uCC28\uB97C \uD574\uC11D\uD558\uACE0 \uC788\uC2B5\uB2C8\uB2E4...",
    executing: "\uC791\uC5C5 \uC808\uCC28\uB97C \uC21C\uCC28\uC801\uC73C\uB85C \uC218\uD589 \uC911\uC785\uB2C8\uB2E4...",
    completed: "\uB3D9\uC791 \uC644\uB8CC\uD588\uC2B5\uB2C8\uB2E4. \uC2E4\uD589 \uB9AC\uD3EC\uD2B8\uB97C \uC0DD\uC131\uD588\uC2B5\uB2C8\uB2E4.",
    executionError: "\uC2E4\uD589 \uC624\uB958\uAC00 \uBC1C\uC0DD\uD588\uC2B5\uB2C8\uB2E4.",
    loadError: "\uC2DC\uC5F0 \uC601\uC0C1\uC744 \uBD88\uB7EC\uC624\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4.",
    missingVideo: "\uC791\uC5C5 \uC601\uC0C1\uC744 \uCC3E\uC744 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4",
    autoplayBlocked: "\uBE0C\uB77C\uC6B0\uC800\uAC00 \uC601\uC0C1 \uC790\uB3D9 \uC7AC\uC0DD\uC744 \uD5C8\uC6A9\uD558\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4",
    playbackError: "\uC601\uC0C1 \uC7AC\uC0DD \uC911 \uC624\uB958\uAC00 \uBC1C\uC0DD\uD588\uC2B5\uB2C8\uB2E4",
    cancelled: "\uC2E4\uD589\uC774 \uCDE8\uC18C\uB418\uC5C8\uC2B5\uB2C8\uB2E4",
    reportDone: "\uB3D9\uC791 \uC644\uB8CC\uD588\uC2B5\uB2C8\uB2E4.",
    reportTitle: "**[\uC2E4\uD589 \uB9AC\uD3EC\uD2B8]**",
    errorLine: "\uC624\uB958: \uC5C6\uC74C",
    mixWarning: "\uD63C\uD569 \uADE0\uC9C8\uB3C4\uB294 \uBCC4\uB3C4 \uC13C\uC11C \uAC80\uC99D \uC5C6\uC774 \uC2DC\uC5F0 \uC601\uC0C1 \uAE30\uC900\uC73C\uB85C \uC644\uB8CC \uCC98\uB9AC\uB418\uC5C8\uC2B5\uB2C8\uB2E4.",
    heatWarning: "\uACE0\uC628 \uC791\uC5C5\uC73C\uB85C \uBD84\uB958\uB418\uC5B4 \uD6C4\uC18D \uC628\uB3C4 \uB85C\uADF8 \uD655\uC778\uC774 \uD544\uC694\uD569\uB2C8\uB2E4.",
};

const LOCAL_VIDEO_PATH = "./%ED%83%91%EB%B7%B0-%EC%8B%9C%EC%97%B0%EC%B5%9C%EC%A2%85.mp4";
const DEMO_SUBGOALS = [
    "Move to source object",
    "Pick source object",
    "Move source to target",
    "Pour source into target",
    "Move source back",
    "Release source object",
    "Move to tool object",
    "Pick tool object",
    "Move tool to target",
    "Stir target",
    "Move tool back",
    "Release tool object"
];

let currentBoxId = null;
let chatHistory = {};
let isExecuting = false;
let activeExecutionToken = 0;
let performanceChartInstance = null;
let isVideoPrimed = false;
let statusPollingInterval = null;

const screenA = document.getElementById("screen-a");
const screenB = document.getElementById("screen-b");
const boxListEl = document.getElementById("box-list");
const btnBack = document.getElementById("btn-back");
const currentBoxNameEl = document.getElementById("current-box-name");
const videoLoaderEl = document.getElementById("video-loader");
const chatHistoryEl = document.getElementById("chat-history");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const typingIndicator = document.getElementById("typing-indicator");
const robotStatusBanner = document.getElementById("robot-status-banner");
const robotStatusText = document.getElementById("robot-status-text");
const btnSnapshot = document.getElementById("btn-snapshot");
const toastContainer = document.getElementById("toast-container");
const videoTimeEl = document.getElementById("video-time");
const btnSettings = document.getElementById("btn-settings");
const settingsModal = document.getElementById("settings-modal");
const btnSettingsCancel = document.getElementById("btn-settings-cancel");
const btnSettingsSave = document.getElementById("btn-settings-save");
const apiKeyInput = document.getElementById("api-key-input");
const btnAutoSequence = document.getElementById("btn-auto-sequence");
const btnPerformance = document.getElementById("btn-performance");
const performanceModal = document.getElementById("performance-modal");
const btnPerformanceClose = document.getElementById("btn-performance-close");
const performanceChartCanvas = document.getElementById("performance-chart-canvas");
const operationVideoEl = document.getElementById("operation-video");
const operationCommandEl = document.getElementById("operation-command");
const operationStateEl = document.getElementById("operation-state");

document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    renderBoxList();
    initChatInputAutosize();
    initOperationVideo();
    updateClock();
    setInterval(updateClock, 1000);
    openDefaultBox();
});

function openDefaultBox() {
    const firstAvailableBox = GLOVE_BOXES.find((box) => box.status !== "offline");
    if (firstAvailableBox) {
        openBox(firstAvailableBox.id);
    }
}

function renderBoxList() {
    boxListEl.innerHTML = "";

    GLOVE_BOXES.forEach((box) => {
        const isOffline = box.status === "offline";
        let statusConfig = { dot: "status-dot-available", text: "Available", bg: "status-bg-available", icon: "check-circle-2" };

        if (box.status === "in-use") {
            statusConfig = { dot: "status-dot-inuse", text: "In Use", bg: "status-bg-inuse", icon: "clock" };
        }

        if (isOffline) {
            statusConfig = { dot: "status-dot-offline", text: "Offline", bg: "status-bg-offline", icon: "alert-circle" };
        }

        const card = document.createElement("div");
        card.className = `box-card bg-surface rounded-2xl p-4 border border-slate-200 shadow-sm flex items-center justify-between ${isOffline ? "opacity-70 cursor-not-allowed" : "cursor-pointer hover:border-primary/50"}`;
        card.innerHTML = `
            <div class="flex items-start gap-4">
                <div class="w-12 h-12 rounded-xl bg-slate-50 flex items-center justify-center border border-slate-100 flex-shrink-0">
                    <i data-lucide="box" class="w-6 h-6 text-slate-400"></i>
                </div>
                <div>
                    <h3 class="font-semibold text-slate-800">${box.name}</h3>
                    <p class="text-xs text-slate-500 mt-0.5">${box.desc}</p>
                    <div class="inline-flex items-center gap-1.5 mt-2 px-2 py-0.5 rounded-md border text-[10px] font-medium tracking-wide ${statusConfig.bg}">
                        <i data-lucide="${statusConfig.icon}" class="w-3 h-3"></i>
                        ${statusConfig.text}
                    </div>
                </div>
            </div>
            <div class="flex flex-col items-end gap-2">
                <div class="w-2.5 h-2.5 rounded-full ${statusConfig.dot}"></div>
                ${!isOffline ? '<i data-lucide="chevron-right" class="w-5 h-5 text-slate-300"></i>' : ""}
            </div>
        `;

        if (!isOffline) {
            card.addEventListener("click", () => openBox(box.id));
        }

        boxListEl.appendChild(card);
    });

    lucide.createIcons({ root: boxListEl });
}

function openBox(boxId) {
    const box = GLOVE_BOXES.find((item) => item.id === boxId);
    if (!box) return;

    currentBoxId = boxId;
    currentBoxNameEl.textContent = box.name;

    if (!chatHistory[boxId]) {
        chatHistory[boxId] = [{ role: "system", content: TEXT.greeting, timestamp: new Date() }];
    }

    screenA.classList.add("hidden");
    screenA.classList.remove("flex");
    screenB.classList.add("flex");
    screenB.classList.remove("hidden");

    initOperationVideo();
    setOperationPanel(TEXT.standby, "Idle");
    renderChatHistory();
}

btnBack.addEventListener("click", () => {
    screenB.classList.add("hidden");
    screenB.classList.remove("flex");
    screenA.classList.add("flex");
    screenA.classList.remove("hidden");

    currentBoxId = null;
    isExecuting = false;
    activeExecutionToken += 1;
    safePauseVideo();
    hideRobotStatus();
    setOperationPanel(TEXT.standby, "Idle");
});

btnSettings.addEventListener("click", () => {
    settingsModal.classList.remove("hidden");
    settingsModal.style.display = "flex";
    requestAnimationFrame(() => {
        settingsModal.classList.remove("opacity-0");
        settingsModal.firstElementChild.classList.remove("scale-95");
    });
    apiKeyInput.value = localStorage.getItem("gemini_api_key") || "";
});

function closeSettings() {
    settingsModal.classList.add("opacity-0");
    settingsModal.firstElementChild.classList.add("scale-95");
    setTimeout(() => {
        settingsModal.classList.add("hidden");
        settingsModal.style.display = "";
    }, 300);
}

btnSettingsCancel.addEventListener("click", closeSettings);

btnSettingsSave.addEventListener("click", () => {
    const key = apiKeyInput.value.trim();
    if (key) {
        localStorage.setItem("gemini_api_key", key);
    } else {
        localStorage.removeItem("gemini_api_key");
    }
    closeSettings();
    showToast("Settings saved", "check-circle-2");
});

btnPerformance.addEventListener("click", () => {
    performanceModal.classList.remove("hidden");
    performanceModal.style.display = "flex";
    requestAnimationFrame(() => {
        performanceModal.classList.remove("opacity-0");
        performanceModal.firstElementChild.classList.remove("scale-95");
    });
    renderPerformanceChart();
});

btnPerformanceClose.addEventListener("click", () => {
    performanceModal.classList.add("opacity-0");
    performanceModal.firstElementChild.classList.add("scale-95");
    setTimeout(() => {
        performanceModal.classList.add("hidden");
        performanceModal.style.display = "";
    }, 300);
});

function renderPerformanceChart() {
    if (performanceChartInstance) {
        performanceChartInstance.destroy();
    }

    const ctx = performanceChartCanvas.getContext("2d");
    performanceChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
            datasets: [{
                label: "Task Success Rate (%)",
                data: [96.5, 97.2, 96.8, 98.1, 98.5, 99.0, 98.4],
                borderColor: "#4f46e5",
                backgroundColor: "rgba(79, 70, 229, 0.1)",
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointBackgroundColor: "#fff",
                pointBorderColor: "#4f46e5",
                pointBorderWidth: 2,
                pointRadius: 4,
            }, {
                label: "Error Rate (%)",
                data: [3.5, 2.8, 3.2, 1.9, 1.5, 1.0, 1.6],
                borderColor: "#e11d48",
                backgroundColor: "transparent",
                borderWidth: 2,
                tension: 0.4,
                borderDash: [5, 5],
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "top",
                    labels: {
                        usePointStyle: true,
                        boxWidth: 8,
                        font: { family: "'Inter', sans-serif", size: 11 }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: "#f1f5f9", drawBorder: false },
                    ticks: { font: { family: "'Inter', sans-serif", size: 10 }, color: "#64748b" }
                },
                x: {
                    grid: { display: false, drawBorder: false },
                    ticks: { font: { family: "'Inter', sans-serif", size: 10 }, color: "#64748b" }
                }
            }
        }
    });
}

btnAutoSequence.addEventListener("click", async () => {
    if (!currentBoxId || isExecuting) return;

    const userMsg = { role: "user", content: TEXT.autoPrompt, timestamp: new Date() };
    chatHistory[currentBoxId].push(userMsg);
    appendMessageUI(userMsg);
    await simulateLLMResponse(TEXT.autoPrompt);
});

function initOperationVideo() {
    // Video is now a live MJPEG img tag from the backend
    isVideoPrimed = true;
}

function handleVideoReady() {
    // Deprecated
}

function revealVideo() {
    // Deprecated
}

function initChatInputAutosize() {
    chatInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = `${this.scrollHeight}px`;
        if (this.value.trim() === "") {
            this.style.height = "auto";
        }
    });

    chatInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });
}

function renderChatHistory() {
    chatHistoryEl.innerHTML = "";
    const messages = chatHistory[currentBoxId] || [];
    messages.forEach((msg) => appendMessageUI(msg));
    scrollToBottom();
}

function appendMessageUI(msg) {
    const isUser = msg.role === "user";
    const isSystemMsg = msg.role === "status";
    const timeStr = new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    const wrapper = document.createElement("div");
    wrapper.className = `chat-message flex flex-col w-full ${isUser ? "items-end" : "items-start"}`;

    if (isSystemMsg) {
        wrapper.className = "chat-message flex justify-center w-full my-1";
        wrapper.innerHTML = `
            <div class="bg-blue-50/80 backdrop-blur border border-blue-100 text-blue-800 text-[11px] px-3 py-1 rounded-full flex items-center gap-1.5 shadow-sm">
                <i data-lucide="info" class="w-3 h-3 text-blue-500"></i>
                <span>${msg.content}</span>
                <span class="text-blue-400 ml-1 text-[9px]">${timeStr}</span>
            </div>
        `;
    } else {
        const bubbleColor = isUser ? "bg-primary text-white rounded-br-sm" : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm shadow-sm";
        wrapper.innerHTML = `
            <div class="flex items-end gap-1.5 max-w-[85%] ${isUser ? "flex-row-reverse" : "flex-row"}">
                ${!isUser ? `
                    <div class="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 border border-blue-200">
                        <i data-lucide="bot" class="w-3.5 h-3.5 text-blue-700"></i>
                    </div>
                ` : ""}
                <div class="px-4 py-2.5 rounded-2xl markdown-body text-[14px] leading-relaxed ${bubbleColor}">
                    ${formatMessageContent(msg.content)}
                </div>
            </div>
            <div class="text-[10px] text-slate-400 mt-1 ${isUser ? "mr-8" : "ml-8"}">${timeStr}</div>
        `;
    }

    chatHistoryEl.appendChild(wrapper);
    lucide.createIcons({ root: wrapper });
    scrollToBottom();
}

function formatMessageContent(text) {
    return text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br/>");
}

function scrollToBottom() {
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = chatInput.value.trim();
    if (!text || !currentBoxId || isExecuting) return;

    const userMsg = { role: "user", content: text, timestamp: new Date() };
    chatHistory[currentBoxId].push(userMsg);
    appendMessageUI(userMsg);

    chatInput.value = "";
    chatInput.style.height = "auto";

    await simulateLLMResponse(text);
});

async function simulateLLMResponse(userText) {
    if (isExecuting) return;

    isExecuting = true;
    const executionToken = ++activeExecutionToken;

    try {
        showTypingIndicator();
        showRobotStatus(TEXT.planning);
        setOperationPanel(userText, "Planning");

        const response = await fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: userText })
        });
        
        if (!response.ok) {
            throw new Error('Failed to send command to robot');
        }

        hideTypingIndicator();
        showRobotStatus("계획 수립 대기 중...");

        return new Promise((resolve, reject) => {
            statusPollingInterval = setInterval(async () => {
                try {
                    const statusRes = await fetch('/api/status');
                    const state = await statusRes.json();
                    
                    if (executionToken !== activeExecutionToken) {
                        clearInterval(statusPollingInterval);
                        reject(new Error(TEXT.cancelled));
                        return;
                    }

                    if (state.status === "running") {
                        showRobotStatus(TEXT.executing);
                        const progressPct = Math.round((state.progress / Math.max(state.total_steps, 1)) * 100);
                        setOperationPanel(state.current_subgoal, `${progressPct}%`);
                    } else if (state.status === "completed") {
                        clearInterval(statusPollingInterval);
                        
                        if (state.report && state.report.plan) {
                            const planStr = state.report.plan.map((step, index) => `${index + 1}. ${step.label}`).join("\n");
                            const planMsg = {
                                role: "system",
                                content: `**[\uC2E4\uD589 \uACC4\uD68D]**\n${planStr}`,
                                timestamp: new Date()
                            };
                            chatHistory[currentBoxId].push(planMsg);
                            appendMessageUI(planMsg);
                        }

                        const reportMsg = {
                            role: "system",
                            content: buildCompletionReport(userText, state.report),
                            timestamp: new Date()
                        };
                        chatHistory[currentBoxId].push(reportMsg);
                        appendMessageUI(reportMsg);

                        showRobotStatus(TEXT.completed);
                        setOperationPanel(userText, "Completed");
                        resolve();
                    } else if (state.status === "error") {
                        clearInterval(statusPollingInterval);
                        reject(new Error(state.report ? state.report.error : "Unknown error"));
                    }

                } catch (e) {
                    console.error("Polling error", e);
                }
            }, 1000);
        });
    } catch (error) {
        if (executionToken === activeExecutionToken) {
            const errorMsg = {
                role: "system",
                content: `\uC791\uC5C5 \uC2E4\uD589 \uC911 \uC624\uB958\uAC00 \uBC1C\uC0DD\uD588\uC2B5\uB2C8\uB2E4.\n- \uC6D0\uC778: ${error.message}`,
                timestamp: new Date()
            };
            chatHistory[currentBoxId].push(errorMsg);
            appendMessageUI(errorMsg);
            showRobotStatus(TEXT.executionError);
            setOperationPanel(userText, "Error");
        }
    } finally {
        hideTypingIndicator();
        if (statusPollingInterval) {
            clearInterval(statusPollingInterval);
            statusPollingInterval = null;
        }
        if (executionToken === activeExecutionToken) {
            isExecuting = false;
        }
    }
}

async function buildExecutionPlan(userText) {
    // Deprecated
    return null;
}

async function playOperationVideo(executionToken) {
    // Deprecated
    return null;
}

function buildCompletionReport(userText, report) {
    const reportLines = [
        TEXT.reportDone,
        "",
        TEXT.reportTitle,
        `- 명령: ${userText}`,
        `- 실행 시간: ${report ? report.execution_time : "N/A"}`,
        `- 성공률: ${report ? report.success_rate : "100%"}`,
        `- 에러 빈도: ${report ? report.error_frequency : "0%"}`,
    ];

    return reportLines.join("\n");
}

function delay(ms) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

function setOperationPanel(command, state) {
    operationCommandEl.textContent = command;
    operationStateEl.textContent = state;
}

function safePauseVideo() {
    if (operationVideoEl) {
        operationVideoEl.pause();
    }
}

function showTypingIndicator() {
    typingIndicator.classList.remove("opacity-0", "translate-y-4");
    typingIndicator.classList.add("opacity-100", "translate-y-0");
}

function hideTypingIndicator() {
    typingIndicator.classList.add("opacity-0", "translate-y-4");
    typingIndicator.classList.remove("opacity-100", "translate-y-0");
}

function showRobotStatus(text) {
    robotStatusText.textContent = text;
    robotStatusBanner.classList.remove("hidden", "h-0", "opacity-0");
    robotStatusBanner.classList.add("h-10", "opacity-100");
}

function hideRobotStatus() {
    robotStatusBanner.classList.add("h-0", "opacity-0");
    robotStatusBanner.classList.remove("h-10", "opacity-100");
    setTimeout(() => {
        if (robotStatusBanner.classList.contains("opacity-0")) {
            robotStatusBanner.classList.add("hidden");
        }
    }, 300);
}

function updateClock() {
    videoTimeEl.textContent = new Date().toLocaleTimeString([], { hour12: false });
}

btnSnapshot.addEventListener("click", () => {
    showToast("Snapshot saved to gallery", "camera");
});

function showToast(message, iconName = "info") {
    const toast = document.createElement("div");
    toast.className = "bg-slate-800 text-white px-4 py-2 rounded-full shadow-lg text-xs font-medium flex items-center gap-2 transform transition-all duration-300 translate-y-[-20px] opacity-0";
    toast.innerHTML = `
        <i data-lucide="${iconName}" class="w-3.5 h-3.5"></i>
        ${message}
    `;

    toastContainer.appendChild(toast);
    lucide.createIcons({ root: toast });

    requestAnimationFrame(() => {
        toast.classList.remove("translate-y-[-20px]", "opacity-0");
        toast.classList.add("translate-y-0", "opacity-100");
    });

    setTimeout(() => {
        toast.classList.remove("translate-y-0", "opacity-100");
        toast.classList.add("translate-y-[-20px]", "opacity-0");
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}
