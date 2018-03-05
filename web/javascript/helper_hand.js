let RED = "#F44336";
let GREEN = "#00E676";
let BLUE = "#304FFE";

class Point {
    constructor(px, py, occluded){
        this.px = px;
        this.py = py;
        this.occluded = occluded;
    }
}

let CURRJOINTCOL = RED;
let OLDJOINTCOL = GREEN;


class HelperHandManager{
    constructor(canvas, joints){
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.joints = joints;
        this.curr_joint_idx = 0;
        this.jointRadius = 10;
        this.redraw();
    }

    redraw(){
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = OLDJOINTCOL;
        for(let i = 0; i < this.curr_joint_idx; i++){
            this.drawjoint(i);
        }
        this.ctx.fillStyle = CURRJOINTCOL;
        this.drawjoint(this.curr_joint_idx);
    }

    drawjoint(idx){
        if(idx < 0 || idx >= this.joints.length)
            return;
        this.ctx.beginPath();
        let x = this.canvas.width * this.joints[idx].px;
        let y = this.canvas.height * this.joints[idx].py;
        this.ctx.arc(x, y, this.jointRadius, 0, 2 * Math.PI);
        this.ctx.fill()
    }

    next(){
        if(this.curr_joint_idx >= this.joints.length)
            return;
        this.curr_joint_idx++;
        this.redraw();
    }

    prev(){
        if (this.curr_joint_idx <= 0)
            return;
        this.curr_joint_idx --;
        this.redraw();
    }
}