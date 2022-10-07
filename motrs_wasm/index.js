const rust = import("./pkg");

rust.then((m) => {
    let mot = m.Mot.new();

    for (let i = 0; i < 10; i ++) {
        let box = [1, 1, 1, 1];
        let det = {
            box,
            score: 1,
            class_id: 1,
            feature: 1,
        }

        mot.step([det])

        let tracks = mot.active_tracks()

        console.log(tracks)
    }
})
