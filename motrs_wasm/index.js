const rust = import("./pkg");

rust.then((m) => {
    let mot = m.MOT.new();
    let box = [1.0, 1.0, 1.0, 1.0];

    for (let i = 0; i < 2; i ++) {
        box = box.map(v => v + 1, box)
        let det = {
            _box: box,
        }

        mot.step([det])
        console.log(i)
        //let tracks = mot.active_tracks()
        //console.log(tracks)
    }
})
