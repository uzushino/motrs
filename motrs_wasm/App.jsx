import { useEffect, useRef, useState } from "react";
import { createPlayground, loadTracker } from "./playground.js";
import {
    FIELDS,
    MOTIONS,
    SHAPES,
    loadSettings,
    readSettings,
    saveSettings,
    validateSettings,
} from "./settings.js";

const SPAWN_HINT = "Click the canvas to place an object, or use Add object.";
const EMPTY_FRAME = {
    frame: 0,
    objectCount: 0,
    hiddenCount: 0,
    detections: [],
    tracks: [],
    colors: {},
    capacityReached: false,
    running: false,
    extraMissing: false,
    detectionsStopped: false,
};

const MotionOptions = () => {
    return MOTIONS.map(([value, label]) => (
        <option key={value} value={value}>
            {label}
        </option>
    ));
};

const SceneSettings = ({
    formRef,
    initialSettings,
    ready,
    message,
    onApply,
    onShuffle,
    onEdit,
}) => {
    return (
        <form id="settings" ref={formRef} onSubmit={onApply} onInput={onEdit}>
            <fieldset
                id="settings-controls"
                className="settings-panel"
                disabled={!ready}
            >
                <legend>Scene settings</legend>
                <div className="settings-grid">
                    <label>
                        Motion
                        <select
                            name="motion"
                            required
                            defaultValue={initialSettings.motion}
                        >
                            <MotionOptions />
                        </select>
                    </label>
                    {FIELDS.map(({ name, label, min, max, step }) => (
                        <label key={name}>
                            {label}
                            <input
                                name={name}
                                type="number"
                                min={min}
                                max={max}
                                step={step}
                                defaultValue={initialSettings[name]}
                                required
                            />
                        </label>
                    ))}
                </div>
                <div
                    className="shape-picker"
                    role="group"
                    aria-label="Obstacle shapes"
                >
                    <strong>Obstacle shapes</strong>
                    {SHAPES.map((shape) => (
                        <label key={shape}>
                            <input
                                name="shapes"
                                type="checkbox"
                                value={shape}
                                defaultChecked={initialSettings.shapes.includes(
                                    shape
                                )}
                            />
                            {shape[0].toUpperCase() + shape.slice(1)}
                        </label>
                    ))}
                </div>
                <p className="settings-note">
                    Positions and sizes vary randomly. Reuse the seed to repeat
                    a scene, or shuffle for a new layout.
                </p>
                <div className="controls">
                    <button className="primary" type="submit">
                        Apply &amp; restart
                    </button>
                    <button id="randomize" type="button" onClick={onShuffle}>
                        Shuffle &amp; restart
                    </button>
                </div>
                <p id="settings-message" role="status">
                    {message}
                </p>
            </fieldset>
        </form>
    );
};

const PlaybackControls = ({
    ready,
    snapshot,
    shapeRef,
    motionRef,
    message,
    onAdd,
    onPause,
    onStep,
    onReplay,
    onExtraMissing,
    onStopDetections,
}) => {
    return (
        <fieldset id="playback" disabled={!ready}>
            <legend>Playback controls</legend>
            <div className="controls">
                <label>
                    New object
                    <select
                        id="spawn-shape"
                        ref={shapeRef}
                        defaultValue="circle"
                    >
                        {["circle", "square", "star", "triangle"].map(
                            (shape) => (
                                <option key={shape} value={shape}>
                                    {shape[0].toUpperCase() + shape.slice(1)}
                                </option>
                            )
                        )}
                    </select>
                </label>
                <label>
                    Motion
                    <select
                        id="spawn-motion"
                        ref={motionRef}
                        defaultValue="wander"
                    >
                        <MotionOptions />
                    </select>
                </label>
                <button
                    id="add-object"
                    type="button"
                    disabled={snapshot.capacityReached}
                    onClick={() => onAdd()}
                >
                    + Add object
                </button>
            </div>
            <p id="spawn-message" className="settings-note" role="status">
                {message}
            </p>
            <div className="controls">
                <button
                    id="pause"
                    type="button"
                    aria-pressed={!snapshot.running}
                    onClick={onPause}
                >
                    {snapshot.running ? "Pause" : "Play"}
                </button>
                <button
                    id="step"
                    type="button"
                    disabled={snapshot.running}
                    onClick={onStep}
                >
                    Step one frame
                </button>
                <button id="reset" type="button" onClick={onReplay}>
                    Replay scene
                </button>
                <label>
                    <input
                        id="dropout"
                        type="checkbox"
                        checked={snapshot.extraMissing}
                        onChange={onExtraMissing}
                    />
                    Extra missed detections
                </label>
                <label>
                    <input
                        id="no-detections"
                        type="checkbox"
                        checked={snapshot.detectionsStopped}
                        onChange={onStopDetections}
                    />
                    Stop all detections
                </label>
            </div>
        </fieldset>
    );
};

const TrackingResults = ({ canvasRef, snapshot, status, error, onPlace }) => {
    const { tracks, detections } = snapshot;
    const counters = [
        ["Frame", "frame", snapshot.frame],
        ["Objects", "ball-count", snapshot.objectCount],
        ["Hidden", "hidden-count", snapshot.hiddenCount],
        ["Detections", "detection-count", detections.length],
        ["Tracks", "track-count", tracks.length],
        [
            "Predicted",
            "predicted-count",
            tracks.filter(({ missed_frames }) => missed_frames > 0).length,
        ],
    ];
    return (
        <>
            <canvas
                ref={canvasRef}
                width="800"
                height="460"
                onClick={onPlace}
                aria-label="Moving objects behind obstacles, with observed and predicted tracking boxes. Track results are also shown in the table below."
            >
                This demo requires a browser with Canvas support.
            </canvas>
            <div className="legend" aria-label="Legend">
                {[
                    ["truth", "Object"],
                    ["obstacle", "Obstacle"],
                    ["detection", "Detection"],
                    ["track", "Observed track"],
                    ["predicted", "Predicted track"],
                ].map(([kind, label]) => (
                    <span key={kind}>
                        <i className={`key ${kind}`} />
                        {label}
                    </span>
                ))}
            </div>
            <div className="stats">
                {counters.map(([label, id, value]) => (
                    <span key={id}>
                        {label} <strong id={id}>{value}</strong>
                    </span>
                ))}
            </div>
            <p id="status" role={error ? "alert" : "status"}>
                {status}
            </p>
            <div className="table-scroll">
                <table aria-label="Current tracking results">
                    <thead>
                        <tr>
                            <th scope="col">Track ID</th>
                            <th scope="col">State</th>
                            <th scope="col">left, top, right, bottom</th>
                        </tr>
                    </thead>
                    <tbody id="tracks">
                        {tracks.length === 0 ? (
                            <tr>
                                <td colSpan={3}>No active tracks</td>
                            </tr>
                        ) : (
                            tracks.map(({ id, _box, missed_frames }) => (
                                <tr key={id}>
                                    <td
                                        title={id}
                                        style={{ color: snapshot.colors[id] }}
                                    >
                                        {id.slice(0, 8)}
                                    </td>
                                    <td>
                                        {missed_frames > 0
                                            ? `Predicted (${missed_frames})`
                                            : "Observed"}
                                    </td>
                                    <td>
                                        {_box
                                            .map((value) => value.toFixed(1))
                                            .join(", ")}
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
            <p className="help">
                Obstacles cover the objects; they do not change their motion.
                motrs receives only visible detections and predicts through
                gaps. Long occlusions, overlaps, or wall bounces can cause an ID
                to change. Pause and step through a scene to inspect what
                happens.
            </p>
            <details>
                <summary>Inspect this frame&apos;s input and output</summary>
                <div className="json">
                    <div>
                        <h2>tracker.step(detections)</h2>
                        <pre id="input">
                            {JSON.stringify(detections, null, 2)}
                        </pre>
                    </div>
                    <div>
                        <h2>tracker.active_tracks()</h2>
                        <pre id="output">{JSON.stringify(tracks, null, 2)}</pre>
                    </div>
                </div>
            </details>
        </>
    );
};

const App = () => {
    const canvasRef = useRef(null);
    const formRef = useRef(null);
    const shapeRef = useRef(null);
    const motionRef = useRef(null);
    const playgroundRef = useRef(null);
    const [initialSettings] = useState(loadSettings);
    const [snapshot, setSnapshot] = useState(EMPTY_FRAME);
    const [ready, setReady] = useState(false);
    const [error, setError] = useState(null);
    const [settingsMessage, setSettingsMessage] = useState("");
    const [spawnMessage, setSpawnMessage] = useState(SPAWN_HINT);

    const fail = (cause) => {
        playgroundRef.current?.dispose();
        playgroundRef.current = null;
        setReady(false);
        setError(cause.message || String(cause));
        console.error(cause);
    };

    useEffect(() => {
        const lifecycle = { cancelled: false, playground: null };
        loadTracker()
            .then((MOT) => {
                // Strict Mode or navigation can clean up while WASM is loading.
                if (lifecycle.cancelled) return;
                lifecycle.playground = createPlayground(
                    canvasRef.current,
                    MOT,
                    setSnapshot,
                    fail
                );
                playgroundRef.current = lifecycle.playground;
                lifecycle.playground.restart(initialSettings);
                saveSettings(initialSettings);
                setSettingsMessage(
                    `Applied: ${initialSettings.ballCount} balls / ${initialSettings.obstacleCount} obstacles / seed ${initialSettings.seed}`
                );
                setReady(true);
            })
            .catch((cause) => {
                if (!lifecycle.cancelled) fail(cause);
            });
        const onPageHide = (event) => {
            if (!event.persisted) lifecycle.playground?.dispose();
        };
        window.addEventListener("pagehide", onPageHide);
        return () => {
            lifecycle.cancelled = true;
            window.removeEventListener("pagehide", onPageHide);
            lifecycle.playground?.dispose();
            if (playgroundRef.current === lifecycle.playground)
                playgroundRef.current = null;
        };
    }, [initialSettings]);

    const run = (action) => {
        if (!playgroundRef.current) return;
        try {
            return action(playgroundRef.current);
        } catch (cause) {
            fail(cause);
        }
    };

    const applySettings = (event) => {
        event.preventDefault();
        if (!formRef.current.reportValidity()) return;
        const value = readSettings(formRef.current);
        const message = validateSettings(value);
        if (message) {
            setSettingsMessage(message);
            return;
        }
        run((playground) => {
            playground.restart(value);
            saveSettings(value);
            setSpawnMessage(SPAWN_HINT);
            setSettingsMessage(
                `Applied: ${value.ballCount} balls / ${value.obstacleCount} obstacles / seed ${value.seed}`
            );
        });
    };

    const addObject = (position) => {
        run((playground) => {
            const shape = shapeRef.current.value;
            const motion = motionRef.current.value;
            const added = playground.addObject(position, shape, motion);
            setSpawnMessage(
                added
                    ? `Added ${shape} with ${motion} motion. Existing tracks are preserved.`
                    : "The scene supports up to 50 objects. Apply settings to start a new scene."
            );
        });
    };

    const placeObject = (event) => {
        if (!ready) return;
        const canvas = event.currentTarget;
        const rect = canvas.getBoundingClientRect();
        addObject({
            x: ((event.clientX - rect.left) * canvas.width) / rect.width,
            y: ((event.clientY - rect.top) * canvas.height) / rect.height,
        });
    };

    const status = error
        ? `Unable to run the demo: ${error}`
        : !ready
        ? "Loading WebAssembly…"
        : snapshot.detectionsStopped
        ? `Detections paused: tracks survive ${snapshot.maxMissedFrames} missed frames, then expire.`
        : "An object is undetected when its center is behind an obstacle. Dashed colored boxes show predicted tracks.";

    return (
        <main>
            <h1>motrs / Tracking playground</h1>
            <p className="intro">
                Add moving objects as the scene runs, try irregular motion, and
                watch motrs predict through obstacles.
            </p>
            <SceneSettings
                formRef={formRef}
                initialSettings={initialSettings}
                ready={ready}
                message={settingsMessage}
                onApply={applySettings}
                onShuffle={() => {
                    formRef.current.elements.seed.value =
                        Math.floor(Math.random() * 2147483646) + 1;
                    formRef.current.requestSubmit();
                }}
                onEdit={() =>
                    setSettingsMessage(
                        "Settings have changed. Click Apply & restart to use them."
                    )
                }
            />
            <PlaybackControls
                ready={ready}
                snapshot={snapshot}
                shapeRef={shapeRef}
                motionRef={motionRef}
                message={spawnMessage}
                onAdd={addObject}
                onPause={() => run((playground) => playground.toggleRunning())}
                onStep={() => run((playground) => playground.step())}
                onReplay={() =>
                    run((playground) => {
                        playground.replay();
                        setSpawnMessage(SPAWN_HINT);
                    })
                }
                onExtraMissing={(event) =>
                    run((playground) =>
                        playground.setExtraMissing(event.target.checked)
                    )
                }
                onStopDetections={(event) =>
                    run((playground) =>
                        playground.setDetectionsStopped(event.target.checked)
                    )
                }
            />
            <TrackingResults
                canvasRef={canvasRef}
                snapshot={snapshot}
                status={status}
                error={error}
                onPlace={placeObject}
            />
        </main>
    );
};

export default App;
