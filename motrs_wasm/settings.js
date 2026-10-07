export const STORAGE_KEY = "motrs.browser.settings.v1";

export const SHAPES = ["square", "star", "circle", "triangle"];
export const MOTIONS = [
    ["straight", "Straight & bounce"],
    ["wander", "Wandering"],
    ["erratic", "Erratic turns"],
];

export const DEFAULTS = {
    ballCount: 5,
    radius: 18,
    minSpeed: 1,
    maxSpeed: 10,
    motion: "straight",
    obstacleCount: 6,
    obstacleSize: 100,
    maxMissedFrames: 40,
    seed: 42,
    shapes: SHAPES,
};

export const FIELDS = [
    { name: "ballCount", label: "Initial balls", min: 0, max: 50, step: 1 },
    { name: "radius", label: "Ball radius (px)", min: 8, max: 30, step: 1 },
    {
        name: "minSpeed",
        label: "Min speed (px / frame)",
        min: 0.5,
        max: 20,
        step: 0.5,
    },
    {
        name: "maxSpeed",
        label: "Max speed (px / frame)",
        min: 0.5,
        max: 20,
        step: 0.5,
    },
    {
        name: "maxMissedFrames",
        label: "Keep missing tracks (frames)",
        min: 1,
        max: 300,
        step: 1,
    },
    { name: "obstacleCount", label: "Obstacles", min: 0, max: 16, step: 1 },
    {
        name: "obstacleSize",
        label: "Obstacle size (px)",
        min: 40,
        max: 160,
        step: 1,
    },
    { name: "seed", label: "Random seed", min: 0, max: 2147483647, step: 1 },
];

export const validateSettings = (value) => {
    for (const { name, label, min, max, step } of FIELDS) {
        const number = value[name];
        if (
            !Number.isFinite(number) ||
            number < min ||
            number > max ||
            (number - min) % step !== 0
        ) {
            return `${label} must be between ${min} and ${max}, in steps of ${step}.`;
        }
    }
    if (!MOTIONS.some(([motion]) => motion === value.motion))
        return "Select a motion pattern.";
    if (value.minSpeed > value.maxSpeed)
        return "Minimum speed must not exceed maximum speed.";
    if (
        !Array.isArray(value.shapes) ||
        value.shapes.some((shape) => !SHAPES.includes(shape))
    )
        return "Select valid obstacle shapes.";
    if (value.obstacleCount > 0 && value.shapes.length === 0)
        return "Select at least one obstacle shape, or set the obstacle count to zero.";
    return null;
};

export const loadSettings = () => {
    try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
        if (!saved || typeof saved !== "object") return DEFAULTS;
        const value = {
            motion: saved.motion ?? DEFAULTS.motion,
            shapes: Array.isArray(saved.shapes)
                ? SHAPES.filter((shape) => saved.shapes.includes(shape))
                : SHAPES,
        };
        for (const { name } of FIELDS)
            value[name] = Number(saved[name] ?? DEFAULTS[name]);
        return validateSettings(value) ? DEFAULTS : value;
    } catch {
        return DEFAULTS;
    }
};

export const readSettings = (form) => {
    const data = new FormData(form);
    const value = { motion: data.get("motion"), shapes: data.getAll("shapes") };
    for (const { name } of FIELDS) value[name] = Number(data.get(name));
    return value;
};

export const saveSettings = (value) => {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    } catch {
        // Storage is optional; playback still works if it is unavailable or full.
    }
};
