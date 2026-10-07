const canvas = document.querySelector("canvas");
const context = canvas.getContext("2d");
const controls = document.querySelector("#playback");
const settingsForm = document.querySelector("#settings");
const settingsControls = document.querySelector("#settings-controls");
const pauseButton = document.querySelector("#pause");
const stepButton = document.querySelector("#step");
const dropout = document.querySelector("#dropout");
const noDetections = document.querySelector("#no-detections");
const status = document.querySelector("#status");
const settingsMessage = document.querySelector("#settings-message");
const rows = document.querySelector("#tracks");
const trails = new Map();
const storageKey = "motrs.browser.settings.v1";
const defaults = {
    ballCount: 5,
    radius: 18,
    minSpeed: 1,
    maxSpeed: 10,
    motion: "straight",
    obstacleCount: 6,
    obstacleSize: 100,
    maxMissedFrames: 40,
    seed: 42,
    shapes: ["square", "star", "circle", "triangle"],
};
let config = { ...defaults };
let balls = [];
let obstacles = [];
let spawnRandom;
let nextColor = 0;
let tracker;
let frame = 0;
let running = false;
let timer;

// A seed reproduces the same balls and occluders after a reload.
function randomGenerator(seed) {
    return () => {
        seed |= 0;
        seed = (seed + 0x6d2b79f5) | 0;
        let value = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}

function obstaclePath(shape, x, y, size, rotation) {
    const path = new Path2D();
    const radius = size / 2;
    if (shape === "circle") {
        path.arc(x, y, radius, 0, Math.PI * 2);
    } else {
        const vertices = shape === "star" ? 10 : shape === "triangle" ? 3 : 4;
        for (let index = 0; index < vertices; index++) {
            const angle = rotation + (index * Math.PI * 2) / vertices;
            const r = shape === "star" && index % 2 ? radius * 0.45 : radius;
            const px = x + r * Math.cos(angle);
            const py = y + r * Math.sin(angle);
            if (index === 0) path.moveTo(px, py);
            else path.lineTo(px, py);
        }
        path.closePath();
    }
    return path;
}

function isOccluded(x, y) {
    return obstacles.some(({ path }) => context.isPointInPath(path, x, y));
}

function generateScene() {
    const random = randomGenerator(config.seed);
    obstacles = Array.from({ length: config.obstacleCount }, (_, index) => {
        const shape = config.shapes[index % config.shapes.length];
        const size = config.obstacleSize * (0.55 + random() * 0.9);
        const margin = size / 2 + 45;
        const x = margin + random() * (canvas.width - margin * 2);
        const y = margin + random() * (canvas.height - margin * 2);
        return {
            shape,
            x,
            y,
            size,
            path: obstaclePath(shape, x, y, size, random() * Math.PI * 2),
        };
    });
    spawnRandom = random;
    balls = Array.from({ length: config.ballCount }, (_, index) =>
        createObject(undefined, "circle", config.motion, index)
    );
}

function createObject(
    position,
    shape = "circle",
    motion = config.motion,
    ordinal = balls.length
) {
    const random = spawnRandom;
    const radius = config.radius * (0.8 + random() * 0.4);
    let x;
    let y;
    if (position) {
        x = Math.max(radius, Math.min(canvas.width - radius, position.x));
        y = Math.max(radius, Math.min(canvas.height - radius, position.y));
    } else {
        for (let attempt = 0; attempt < 300; attempt++) {
            x = radius + random() * (canvas.width - radius * 2);
            y = radius + random() * (canvas.height - radius * 2);
            if (
                ![
                    [x, y],
                    [x - radius, y],
                    [x + radius, y],
                    [x, y - radius],
                    [x, y + radius],
                ].some(([px, py]) => isOccluded(px, py))
            )
                break;
        }
    }
    const angle = random() * Math.PI * 2;
    const speed =
        config.minSpeed + random() * (config.maxSpeed - config.minSpeed);
    const initial = {
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
    };
    const object = {
        initial,
        radius,
        shape,
        motion,
        baseSpeed: speed,
        bornAt: frame,
        trajectorySeed:
            (config.seed + Math.imul(ordinal + 1, 0x9e3779b9)) >>> 0,
    };
    resetObject(object);
    return object;
}

function resetObject(object) {
    Object.assign(object, object.initial);
    object.motionRandom = randomGenerator(object.trajectorySeed);
    object.simulatedFrame = object.bornAt;
    object.targetAngle = Math.atan2(object.vy, object.vx);
    object.targetSpeed = Math.hypot(object.vx, object.vy);
    object.turnIn = 0;
}

function moveObject(object) {
    if (object.motion !== "straight") {
        const random = object.motionRandom;
        let angle = Math.atan2(object.vy, object.vx);
        let speed = Math.hypot(object.vx, object.vy);
        if (object.turnIn-- <= 0) {
            const turn = object.motion === "erratic" ? 2.6 : 1.8;
            object.targetAngle = angle + (random() - 0.5) * turn;
            object.targetSpeed = Math.max(
                config.minSpeed,
                Math.min(
                    config.maxSpeed,
                    object.baseSpeed * (0.65 + random() * 0.7)
                )
            );
            object.turnIn = 8 + Math.floor(random() * 20);
        }
        const difference = Math.atan2(
            Math.sin(object.targetAngle - angle),
            Math.cos(object.targetAngle - angle)
        );
        angle += difference * (object.motion === "erratic" ? 1 : 0.12);
        speed += (object.targetSpeed - speed) * 0.12;
        object.vx = Math.cos(angle) * speed;
        object.vy = Math.sin(angle) * speed;
    }
    object.x += object.vx;
    object.y += object.vy;
    const min = object.radius;
    const maxX = canvas.width - min;
    const maxY = canvas.height - min;
    if (object.x < min || object.x > maxX) {
        object.x = object.x < min ? min * 2 - object.x : maxX * 2 - object.x;
        object.vx = -object.vx;
        object.targetAngle = Math.PI - object.targetAngle;
    }
    if (object.y < min || object.y > maxY) {
        object.y = object.y < min ? min * 2 - object.y : maxY * 2 - object.y;
        object.vy = -object.vy;
        object.targetAngle = -object.targetAngle;
    }
    object.simulatedFrame += 1;
}

// Scene randomness stays local; only bounding boxes reach the Rust tracker.
// One tracking frame represents 0.1 seconds, matching the Rust model's dt.
function scene(frameNumber) {
    return balls
        .filter(({ bornAt }) => bornAt <= frameNumber)
        .map((object) => {
            while (object.simulatedFrame < frameNumber) moveObject(object);
            const { x, y, radius, shape } = object;
            return {
                x,
                y,
                radius,
                shape,
                hidden: isOccluded(x, y),
                bounds: [x - radius, y - radius, x + radius, y + radius],
            };
        });
}

function rectangle(bounds, color, dashed = false) {
    const [left, top, right, bottom] = bounds;
    context.strokeStyle = color;
    context.lineWidth = dashed ? 1.5 : 2;
    context.setLineDash(dashed ? [5, 4] : []);
    context.strokeRect(left, top, right - left, bottom - top);
    context.setLineDash([]);
}

function draw(objects, detections, tracks) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#edf0f5";
    context.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, canvas.height);
        context.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(canvas.width, y);
        context.stroke();
    }
    objects.forEach(({ x, y, radius, shape }) => {
        if (shape !== "circle") {
            context.fillStyle = "#cbd5e1";
            context.fill(obstaclePath(shape, x, y, radius * 2, -Math.PI / 2));
            return;
        }
        context.fillStyle = "#cbd5e1";
        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = "#f8fafc";
        context.beginPath();
        context.arc(
            x - radius * 0.3,
            y - radius * 0.3,
            radius * 0.22,
            0,
            Math.PI * 2
        );
        context.fill();
    });
    // Occluders cover the balls; predicted tracks remain visible above them.
    obstacles.forEach(({ path }) => {
        context.fillStyle = "#64748b";
        context.strokeStyle = "#475569";
        context.lineWidth = 1;
        context.fill(path);
        context.stroke(path);
    });
    detections.forEach(({ _box }) => rectangle(_box, "#334155", true));
    tracks.forEach(({ id, _box, missed_frames }) => {
        const trail = trails.get(id);
        context.strokeStyle = trail.color;
        context.lineWidth = 1.5;
        context.beginPath();
        trail.points.forEach(([x, y], index) => {
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
        });
        context.stroke();
        rectangle(_box, trail.color, missed_frames > 0);
        const label = `${id.slice(0, 8)}${
            missed_frames > 0 ? " · predicted" : ""
        }`;
        context.font = "12px ui-monospace, monospace";
        const labelWidth = context.measureText(label).width;
        const labelX = Math.max(
            2,
            Math.min(_box[0], canvas.width - labelWidth - 6)
        );
        const labelY = Math.max(16, Math.min(_box[1] - 7, canvas.height - 4));
        context.fillStyle = "rgba(255, 255, 255, 0.92)";
        context.fillRect(labelX - 2, labelY - 13, labelWidth + 4, 17);
        context.fillStyle = trail.color;
        context.fillText(label, labelX, labelY);
    });
}

function showTracks(tracks) {
    rows.replaceChildren();
    tracks.forEach(({ id, _box, missed_frames }) => {
        const row = document.createElement("tr");
        const idCell = document.createElement("td");
        idCell.textContent = id.slice(0, 8);
        idCell.title = id;
        idCell.style.color = trails.get(id).color;
        const stateCell = document.createElement("td");
        stateCell.textContent =
            missed_frames > 0 ? `Predicted (${missed_frames})` : "Observed";
        const boundsCell = document.createElement("td");
        boundsCell.textContent = _box
            .map((value) => value.toFixed(1))
            .join(", ");
        row.append(idCell, stateCell, boundsCell);
        rows.append(row);
    });
    if (tracks.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 3;
        cell.textContent = "No active tracks";
        row.append(cell);
        rows.append(row);
    }
}

function advance() {
    const objects = scene(frame);
    const missing = dropout.checked && frame % 60 >= 40 && frame % 60 < 48;
    const dropoutIndex = Math.min(1, objects.length - 1);
    const detections = noDetections.checked
        ? []
        : objects
              .filter(
                  (object, index) =>
                      !object.hidden && !(index === dropoutIndex && missing)
              )
              .map(({ bounds }) => ({ _box: bounds }));
    tracker.step(detections);
    const tracks = tracker.active_tracks();
    const activeIds = new Set(tracks.map(({ id }) => id));
    for (const id of trails.keys()) {
        if (!activeIds.has(id)) trails.delete(id);
    }
    tracks.forEach(({ id, _box: [left, top, right, bottom] }) => {
        if (!trails.has(id)) {
            trails.set(id, {
                color: `hsl(${(nextColor++ * 137.508 + 215) % 360}, 75%, 42%)`,
                points: [],
            });
        }
        const points = trails.get(id).points;
        points.push([(left + right) / 2, (top + bottom) / 2]);
        if (points.length > 50) points.shift();
    });
    frame += 1;
    draw(objects, detections, tracks);
    showTracks(tracks);
    document.querySelector("#frame").textContent = frame;
    document.querySelector("#ball-count").textContent = objects.length;
    document.querySelector("#add-object").disabled = balls.length >= 50;
    document.querySelector("#hidden-count").textContent = objects.filter(
        ({ hidden }) => hidden
    ).length;
    document.querySelector("#detection-count").textContent = detections.length;
    document.querySelector("#track-count").textContent = tracks.length;
    document.querySelector("#predicted-count").textContent = tracks.filter(
        ({ missed_frames }) => missed_frames > 0
    ).length;
    document.querySelector("#input").textContent = JSON.stringify(
        detections,
        null,
        2
    );
    document.querySelector("#output").textContent = JSON.stringify(
        tracks,
        null,
        2
    );
    status.textContent = noDetections.checked
        ? `Detections paused: tracks survive ${config.maxMissedFrames} missed frames, then expire.`
        : "An object is undetected when its center is behind an obstacle. Dashed colored boxes show predicted tracks.";
}

function setRunning(value) {
    running = value;
    pauseButton.textContent = running ? "Pause" : "Play";
    pauseButton.setAttribute("aria-pressed", String(!running));
    stepButton.disabled = running;
}

function fail(error) {
    clearInterval(timer);
    setRunning(false);
    controls.disabled = true;
    settingsControls.disabled = true;
    status.textContent = `Unable to run the demo: ${error.message || error}`;
    status.setAttribute("role", "alert");
    console.error(error);
}

function safely(action) {
    try {
        action();
    } catch (error) {
        fail(error);
    }
}

function populateSettings(value) {
    for (const key of Object.keys(defaults).filter((key) => key !== "shapes")) {
        settingsForm.elements[key].value = value[key] ?? defaults[key];
    }
    const shapes = Array.isArray(value.shapes) ? value.shapes : defaults.shapes;
    settingsForm.querySelectorAll('[name="shapes"]').forEach((input) => {
        input.checked = shapes.includes(input.value);
    });
}

function readSettings() {
    if (!settingsForm.reportValidity()) return null;
    const value = {};
    for (const key of Object.keys(defaults).filter(
        (key) => typeof defaults[key] === "number"
    )) {
        value[key] = Number(settingsForm.elements[key].value);
    }
    value.motion = settingsForm.elements.motion.value;
    if (value.minSpeed > value.maxSpeed) {
        settingsMessage.textContent =
            "Minimum speed must not exceed maximum speed.";
        return null;
    }
    value.shapes = Array.from(
        settingsForm.querySelectorAll('[name="shapes"]:checked'),
        ({ value }) => value
    );
    if (value.obstacleCount > 0 && value.shapes.length === 0) {
        settingsMessage.textContent =
            "Select at least one obstacle shape, or set the obstacle count to zero.";
        return null;
    }
    return value;
}

try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    populateSettings(saved && typeof saved === "object" ? saved : defaults);
    // Enable validation briefly; disabled fieldsets skip native validation.
    settingsControls.disabled = false;
    if (
        !settingsForm.checkValidity() ||
        Number(settingsForm.elements.minSpeed.value) >
            Number(settingsForm.elements.maxSpeed.value) ||
        (Number(settingsForm.elements.obstacleCount.value) > 0 &&
            !settingsForm.querySelector('[name="shapes"]:checked'))
    ) {
        populateSettings(defaults);
    }
} catch {
    populateSettings(defaults);
} finally {
    settingsControls.disabled = true;
}

import("./pkg")
    .then(async ({ default: init, MOT }) => {
        await init();
        function reset() {
            if (tracker) tracker.free();
            tracker = MOT.with_max_missed_frames(config.maxMissedFrames);
            trails.clear();
            nextColor = 0;
            frame = 0;
            noDetections.checked = false;
            balls.forEach(resetObject);
            document.querySelector("#spawn-message").textContent =
                "Click the canvas to place an object, or use Add object.";
            advance();
        }
        function applySettings() {
            const value = readSettings();
            if (!value) return;
            config = value;
            try {
                localStorage.setItem(storageKey, JSON.stringify(config));
            } catch {
                /* Storage is optional. */
            }
            frame = 0;
            balls = [];
            generateScene();
            reset();
            setRunning(true);
            settingsMessage.textContent = `Applied: ${config.ballCount} balls / ${config.obstacleCount} obstacles / seed ${config.seed}`;
        }
        controls.disabled = false;
        settingsControls.disabled = false;
        applySettings();
        settingsForm.addEventListener("submit", (event) => {
            event.preventDefault();
            safely(applySettings);
        });
        document.querySelector("#randomize").addEventListener("click", () => {
            settingsForm.elements.seed.value =
                Math.floor(Math.random() * 2147483646) + 1;
            safely(applySettings);
        });
        settingsForm.addEventListener("input", () => {
            settingsMessage.textContent =
                "Settings have changed. Click Apply & restart to use them.";
        });
        function addObject(position) {
            const message = document.querySelector("#spawn-message");
            if (balls.length >= 50) {
                message.textContent =
                    "The scene supports up to 50 objects. Apply settings to start a new scene.";
                return;
            }
            const shape = document.querySelector("#spawn-shape").value;
            const motion = document.querySelector("#spawn-motion").value;
            balls.push(createObject(position, shape, motion));
            // Process the new detection without recreating the tracker.
            advance();
            message.textContent = `Added ${shape} with ${motion} motion. Existing tracks are preserved.`;
        }
        document
            .querySelector("#add-object")
            .addEventListener("click", () => safely(() => addObject()));
        canvas.addEventListener("click", (event) => {
            if (controls.disabled) return;
            const rect = canvas.getBoundingClientRect();
            safely(() =>
                addObject({
                    x:
                        ((event.clientX - rect.left) * canvas.width) /
                        rect.width,
                    y:
                        ((event.clientY - rect.top) * canvas.height) /
                        rect.height,
                })
            );
        });
        pauseButton.addEventListener("click", () => setRunning(!running));
        stepButton.addEventListener("click", () => safely(advance));
        document
            .querySelector("#reset")
            .addEventListener("click", () => safely(reset));
        timer = setInterval(() => {
            if (running && !document.hidden) safely(advance);
        }, 100);
        window.addEventListener("pagehide", (event) => {
            if (!event.persisted) {
                clearInterval(timer);
                tracker.free();
            }
        });
    })
    .catch(fail);
