let RED = "#F44336";
let GREEN = "#4CAF50";
let YELLOW = "#FDD835";

class Point {
    constructor(px, py, occluded){
        this.px = px;
        this.py = py;
        this.occluded = occluded;
    }
}

class Points {
    constructor(l_canvas, r_canvas, target_joints) {
        this.N_POINTS = 21;
        // Pixel width
        this.PW = 10;
        this.selected_points = [];

        this.l_canvas = l_canvas;
        this.r_canvas = r_canvas;
        this.target_joints = target_joints;
        this.l_ctx = l_canvas.getContext("2d");
        this.r_ctx = r_canvas.getContext("2d");
        this.l_ctx.fillStyle=RED;

        console.log(target_joints);

        this.current_point = this.target_joints[0];
        this.r_ctx.fillStyle = YELLOW;
        this.r_ctx.fillRect(this.current_point.px * this.l_canvas.width,
            this.current_point.py * this.l_canvas.height, this.PW, this.PW);

    }

    add_point(px, py, occluded) {
        occluded ? this.l_ctx.fillStyle = GREEN : this.l_ctx.fillStyle = RED;
        if (this.selected_points.length < this.N_POINTS) {
            this.selected_points.push(new Point(px, py, occluded));
            this.l_ctx.fillRect(px * this.l_canvas.width,
                py * this.l_canvas.height, this.PW, this.PW);
        }

        // change color to current point
        this.r_ctx.fillStyle = GREEN;
        this.r_ctx.fillRect(this.current_point.px * this.l_canvas.width,
            this.current_point.py * this.l_canvas.height, this.PW, this.PW);

        // Add new target point
        if (this.selected_points.length < this.N_POINTS) {
            this.r_ctx.fillStyle = YELLOW;
            this.current_point = this.target_joints[this.selected_points.length];
            this.r_ctx.fillRect(this.current_point.px * this.r_canvas.width,
                this.current_point.py * this.r_canvas.height, this.PW, this.PW);
        }

        this.print_on_console();
    }

    undo() {
        if(this.selected_points.length > 0) {
            if(this.selected_points.length < this.N_POINTS){
                var to_delete = this.target_joints[this.selected_points.length];
                this.r_ctx.clearRect(to_delete.px * this.r_canvas.width,
                    to_delete.py * this.r_canvas.height, this.PW, this.PW);
            }

            var point = this.selected_points.pop();
            this.l_ctx.clearRect(point.px * this.l_canvas.width,
                point.py * this.l_canvas.height, this.PW, this.PW);

            this.r_ctx.fillStyle = YELLOW;
            this.current_point = this.target_joints[this.selected_points.length];
            this.r_ctx.fillRect(this.current_point.px * this.r_canvas.width,
                this.current_point.py * this.r_canvas.height, this.PW, this.PW);
        }
    }

    reset() {
        while(this.selected_points.length > 0)
            this.undo();
    }

    print_on_console() {
        var log = "";
        for (var i = 0; i < this.selected_points.length; i++)
            log = log + "px: " + this.selected_points[i].px + ", py: "
                + this.selected_points[i].py + '\n';
        console.log("ArrayLength = " + this.selected_points.length);
        console.log(log);
    }

}