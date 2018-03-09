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
        this.jointRadius = 7;
        this.redraw();
    }

    redraw(){
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawsegments();
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 1;
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
        this.ctx.fill();
        this.ctx.stroke();
    }

    drawsegments(){
        this.ctx.lineWidth = 2;
        for(let jointindex=0; jointindex<this.joints.length; jointindex++) {
            let todolist = SEGMENTS_LIST[jointindex];
            for (let i = 0; i < todolist.length; i++) {
                this.ctx.beginPath();
                console.log(i);
                this.ctx.strokeStyle = todolist[i][1];
                let p1 = this.joints[todolist[i][0]];
                let p2 = this.joints[jointindex];
                this.ctx.moveTo(p1.px * this.canvas.width, p1.py * this.canvas.height);
                this.ctx.lineTo(p2.px * this.canvas.width, p2.py * this.canvas.height);
                this.ctx.stroke();
            }
        }
    }

    reset(){
        this.curr_joint_idx = 0;
        this.redraw();
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