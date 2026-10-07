import { DEFAULTS } from "./settings.js";

const wasmLoader = { promise: null };

// Share WASM initialization across React effect mounts and hot reloads.
export const loadTracker = () => {
    wasmLoader.promise ??= import("./pkg/index.js").then(
        async ({ default: init, MOT }) => {
            await init();
            return MOT;
        }
    );
    return wasmLoader.promise;
};

// Canvas and tracker state have one lifetime, independent of React renders.
export const createPlayground = (canvas, MOT, onFrame, onError) => {
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D rendering is unavailable.");
    const trails = new Map();
    const state = {
        config: { ...DEFAULTS },
        balls: [],
        obstacles: [],
        spawnRandom: null,
        nextColor: 0,
        tracker: null,
        frame: 0,
        running: false,
        extraMissing: false,
        detectionsStopped: false,
        disposed: false,
        snapshot: {
            frame: 0,
            objectCount: 0,
            hiddenCount: 0,
            detections: [],
            tracks: [],
            colors: {},
            capacityReached: false,
        },
    };

    const publish = () => {
        onFrame({
            ...state.snapshot,
            running: state.running,
            extraMissing: state.extraMissing,
            detectionsStopped: state.detectionsStopped,
            maxMissedFrames: state.config.maxMissedFrames,
        });
    };

    // A seed reproduces the same balls and occluders after a reload.
    const randomGenerator = (seed) => {
        const randomState = { seed };
        return () => {
            randomState.seed = ((randomState.seed | 0) + 0x6d2b79f5) | 0;
            const value = Math.imul(
                randomState.seed ^ (randomState.seed >>> 15),
                1 | randomState.seed
            );
            const mixed =
                value ^ (value + Math.imul(value ^ (value >>> 7), 61 | value));
            return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
        };
    };

    const obstaclePath = (shape, x, y, size, rotation) => {
        const path = new Path2D();
        const radius = size / 2;
        if (shape === "circle") {
            path.arc(x, y, radius, 0, Math.PI * 2);
        } else {
            const vertices =
                shape === "star" ? 10 : shape === "triangle" ? 3 : 4;
            for (const index of Array.from({ length: vertices }, (_, i) => i)) {
                const angle = rotation + (index * Math.PI * 2) / vertices;
                const r =
                    shape === "star" && index % 2 ? radius * 0.45 : radius;
                const px = x + r * Math.cos(angle);
                const py = y + r * Math.sin(angle);
                if (index === 0) path.moveTo(px, py);
                else path.lineTo(px, py);
            }
            path.closePath();
        }
        return path;
    };

    const isOccluded = (x, y) => {
        return state.obstacles.some(({ path }) =>
            context.isPointInPath(path, x, y)
        );
    };

    const generateScene = () => {
        const random = randomGenerator(state.config.seed);
        state.obstacles = Array.from(
            { length: state.config.obstacleCount },
            (_, index) => {
                const shape =
                    state.config.shapes[index % state.config.shapes.length];
                const size =
                    state.config.obstacleSize * (0.55 + random() * 0.9);
                const margin = size / 2 + 45;
                const x = margin + random() * (canvas.width - margin * 2);
                const y = margin + random() * (canvas.height - margin * 2);
                return {
                    shape,
                    x,
                    y,
                    size,
                    path: obstaclePath(
                        shape,
                        x,
                        y,
                        size,
                        random() * Math.PI * 2
                    ),
                };
            }
        );
        state.spawnRandom = random;
        state.balls = Array.from(
            { length: state.config.ballCount },
            (_, index) =>
                createObject(undefined, "circle", state.config.motion, index)
        );
    };

    const createObject = (
        position,
        shape = "circle",
        motion = state.config.motion,
        ordinal = state.balls.length
    ) => {
        const random = state.spawnRandom;
        const radius = state.config.radius * (0.8 + random() * 0.4);
        const point = { x: 0, y: 0 };
        if (position) {
            point.x = Math.max(
                radius,
                Math.min(canvas.width - radius, position.x)
            );
            point.y = Math.max(
                radius,
                Math.min(canvas.height - radius, position.y)
            );
        } else {
            Array.from({ length: 300 }).some(() => {
                point.x = radius + random() * (canvas.width - radius * 2);
                point.y = radius + random() * (canvas.height - radius * 2);
                return ![
                    [point.x, point.y],
                    [point.x - radius, point.y],
                    [point.x + radius, point.y],
                    [point.x, point.y - radius],
                    [point.x, point.y + radius],
                ].some(([px, py]) => isOccluded(px, py));
            });
        }
        const angle = random() * Math.PI * 2;
        const speed =
            state.config.minSpeed +
            random() * (state.config.maxSpeed - state.config.minSpeed);
        const initial = {
            x: point.x,
            y: point.y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
        };
        const object = {
            initial,
            radius,
            shape,
            motion,
            baseSpeed: speed,
            bornAt: state.frame,
            trajectorySeed:
                (state.config.seed + Math.imul(ordinal + 1, 0x9e3779b9)) >>> 0,
        };
        resetObject(object);
        return object;
    };

    const resetObject = (object) => {
        Object.assign(object, object.initial);
        object.motionRandom = randomGenerator(object.trajectorySeed);
        object.simulatedFrame = object.bornAt;
        object.targetAngle = Math.atan2(object.vy, object.vx);
        object.targetSpeed = Math.hypot(object.vx, object.vy);
        object.turnIn = 0;
    };

    const moveObject = (object) => {
        if (object.motion !== "straight") {
            const random = object.motionRandom;
            const angle = Math.atan2(object.vy, object.vx);
            const speed = Math.hypot(object.vx, object.vy);
            if (object.turnIn-- <= 0) {
                const turn = object.motion === "erratic" ? 2.6 : 1.8;
                object.targetAngle = angle + (random() - 0.5) * turn;
                object.targetSpeed = Math.max(
                    state.config.minSpeed,
                    Math.min(
                        state.config.maxSpeed,
                        object.baseSpeed * (0.65 + random() * 0.7)
                    )
                );
                object.turnIn = 8 + Math.floor(random() * 20);
            }
            const difference = Math.atan2(
                Math.sin(object.targetAngle - angle),
                Math.cos(object.targetAngle - angle)
            );
            const nextAngle =
                angle + difference * (object.motion === "erratic" ? 1 : 0.12);
            const nextSpeed = speed + (object.targetSpeed - speed) * 0.12;
            object.vx = Math.cos(nextAngle) * nextSpeed;
            object.vy = Math.sin(nextAngle) * nextSpeed;
        }
        object.x += object.vx;
        object.y += object.vy;
        const min = object.radius;
        const maxX = canvas.width - min;
        const maxY = canvas.height - min;
        if (object.x < min || object.x > maxX) {
            object.x =
                object.x < min ? min * 2 - object.x : maxX * 2 - object.x;
            object.vx = -object.vx;
            object.targetAngle = Math.PI - object.targetAngle;
        }
        if (object.y < min || object.y > maxY) {
            object.y =
                object.y < min ? min * 2 - object.y : maxY * 2 - object.y;
            object.vy = -object.vy;
            object.targetAngle = -object.targetAngle;
        }
        object.simulatedFrame += 1;
    };

    // Scene randomness stays local; only bounding boxes reach the Rust tracker.
    // One tracking frame represents 0.1 seconds, matching the Rust model's dt.
    const scene = (frameNumber) => {
        return state.balls
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
    };

    const rectangle = (bounds, color, dashed = false) => {
        const [left, top, right, bottom] = bounds;
        context.strokeStyle = color;
        context.lineWidth = dashed ? 1.5 : 2;
        context.setLineDash(dashed ? [5, 4] : []);
        context.strokeRect(left, top, right - left, bottom - top);
        context.setLineDash([]);
    };

    const draw = (objects, detections, tracks) => {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.strokeStyle = "#edf0f5";
        context.lineWidth = 1;
        for (const x of Array.from(
            { length: Math.ceil(canvas.width / 40) },
            (_, index) => index * 40
        )) {
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, canvas.height);
            context.stroke();
        }
        for (const y of Array.from(
            { length: Math.ceil(canvas.height / 40) },
            (_, index) => index * 40
        )) {
            context.beginPath();
            context.moveTo(0, y);
            context.lineTo(canvas.width, y);
            context.stroke();
        }
        objects.forEach(({ x, y, radius, shape }) => {
            if (shape !== "circle") {
                context.fillStyle = "#cbd5e1";
                context.fill(
                    obstaclePath(shape, x, y, radius * 2, -Math.PI / 2)
                );
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
        state.obstacles.forEach(({ path }) => {
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
            const labelY = Math.max(
                16,
                Math.min(_box[1] - 7, canvas.height - 4)
            );
            context.fillStyle = "rgba(255, 255, 255, 0.92)";
            context.fillRect(labelX - 2, labelY - 13, labelWidth + 4, 17);
            context.fillStyle = trail.color;
            context.fillText(label, labelX, labelY);
        });
    };

    const advance = () => {
        const objects = scene(state.frame);
        const missing =
            state.extraMissing &&
            state.frame % 60 >= 40 &&
            state.frame % 60 < 48;
        const dropoutIndex = Math.min(1, objects.length - 1);
        const detections = state.detectionsStopped
            ? []
            : objects
                  .filter(
                      (object, index) =>
                          !object.hidden && !(index === dropoutIndex && missing)
                  )
                  .map(({ bounds }) => ({ _box: bounds }));
        state.tracker.step(detections);
        const tracks = state.tracker.active_tracks();
        const activeIds = new Set(tracks.map(({ id }) => id));
        for (const id of trails.keys()) {
            if (!activeIds.has(id)) trails.delete(id);
        }
        tracks.forEach(({ id, _box: [left, top, right, bottom] }) => {
            if (!trails.has(id)) {
                trails.set(id, {
                    color: `hsl(${
                        (state.nextColor++ * 137.508 + 215) % 360
                    }, 75%, 42%)`,
                    points: [],
                });
            }
            const points = trails.get(id).points;
            points.push([(left + right) / 2, (top + bottom) / 2]);
            if (points.length > 50) points.shift();
        });
        state.frame += 1;
        draw(objects, detections, tracks);
        state.snapshot = {
            frame: state.frame,
            objectCount: objects.length,
            hiddenCount: objects.filter(({ hidden }) => hidden).length,
            detections,
            tracks,
            colors: Object.fromEntries(
                [...trails].map(([id, trail]) => [id, trail.color])
            ),
            capacityReached: state.balls.length >= 50,
        };
        publish();
    };

    const setRunning = (value) => {
        state.running = value;
        publish();
    };

    const replay = () => {
        if (state.disposed) return;
        if (state.tracker) state.tracker.free();
        state.tracker = MOT.with_max_missed_frames(
            state.config.maxMissedFrames
        );
        trails.clear();
        state.nextColor = 0;
        state.frame = 0;
        state.detectionsStopped = false;
        state.balls.forEach(resetObject);
        advance();
    };

    const restart = (value) => {
        state.config = { ...value, shapes: [...value.shapes] };
        state.frame = 0;
        state.balls = [];
        generateScene();
        replay();
        setRunning(true);
    };

    const addObject = (position, shape, motion) => {
        if (state.balls.length >= 50) return false;
        state.balls.push(createObject(position, shape, motion));
        // Process the new observation without replacing any existing tracker.
        advance();
        return true;
    };

    const timer = setInterval(() => {
        if (!state.disposed && state.running && !document.hidden) {
            try {
                advance();
            } catch (error) {
                dispose();
                onError(error);
            }
        }
    }, 100);

    const dispose = () => {
        if (state.disposed) return;
        state.disposed = true;
        state.running = false;
        clearInterval(timer);
        if (state.tracker) state.tracker.free();
        state.tracker = null;
        trails.clear();
    };

    return {
        restart,
        replay,
        step: advance,
        toggleRunning: () => setRunning(!state.running),
        addObject,
        setExtraMissing: (value) => {
            state.extraMissing = value;
            publish();
        },
        setDetectionsStopped: (value) => {
            state.detectionsStopped = value;
            publish();
        },
        dispose,
    };
};
