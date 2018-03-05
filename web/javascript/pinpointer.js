let VISIBLEJOINTCOL = GREEN;
let OCCLUDEDJOINTCOL = BLUE;
let SELECTIONRECTCOL = RED;

let SELRECTTHRESH = 0.08;

class Pinpointer{
    constructor(front_canvas, bkg_canvas, bkgimgurl){
        this.front_canvas = front_canvas;
        this.front_ctx = this.front_canvas.getContext("2d");
        this.front_canvas.onmousedown = (e) => this.clicked(e);
        this.front_canvas.onmousemove = (e) => this.selector_follow(e);
        this.front_canvas.onmouseup = (e) => this.determine_action(e);
        document.addEventListener("keyup", (e) => this.resetBB(e));

        this.bkg_canvas = bkg_canvas;
        this.bkg_ctx = this.bkg_canvas.getContext("2d");
        this.bkgurl = bkgimgurl;
        this.bkgimg = new Image();
        this.bkgimg.onload = () => this.drawbkg();
        this.bkgimg.src = this.bkgurl;

        this.joints = [];
        this.jointRadius = 7;
        this.current_bb_x = [0, 1];
        this.current_bb_y = [0, 1];

        this.attention_state = false;
        this.selector_rectangle_is_out = false;
        this.attention_start_pnt = null;
        this.attention_curr_pnt = null;
    }

    resetBB(e){
        if(e.which === 27) {
            this.current_bb_x = [0, 1];
            this.current_bb_y = [0, 1];
            this.drawbkg();
            this.drawfront();
        }
    }

    right_clicked(e){
        let relcoords = getRelativeCoords(e);
        let globrel = this.fromBBoxRelToGlobalRelCoords(relcoords);
        this.setNewJoint(globrel, true);
        this.drawfront();
    }

    clicked(e){
        if(e.button === 0)
            this.set_attention_state(e);
    }

    set_attention_state(e){
        this.attention_state = true;
        this.attention_start_pnt = getRelativeCoords(e);
    }

    selector_follow(e){
        if(this.attention_state) {
            this.attention_curr_pnt = getRelativeCoords(e);
            if(this.shouldConsiderZoomSelection()) {
                this.selector_rectangle_is_out = true;
                this.drawfront();
            }else if (this.selector_rectangle_is_out){
                this.drawfront();
                this.selector_rectangle_is_out = false;
            }
        }
    }

    determine_action(e){
        if(e.button === 2) {
            this.right_clicked(e);
            return;
        }
        else if (e.button === 0) {
            this.attention_curr_pnt = getRelativeCoords(e);
            if (this.shouldConsiderZoomSelection()) {
                let absstart = this.fromBBoxRelToGlobalRelCoords(this.attention_start_pnt);
                let absend = this.fromBBoxRelToGlobalRelCoords(this.attention_curr_pnt);
                this.current_bb_x = [Math.min(absstart[0], absend[0]), Math.max(absstart[0], absend[0])];
                this.current_bb_y = [Math.min(absstart[1], absend[1]), Math.max(absstart[1], absend[1])];
            }
            else {
                let globcoords = this.fromBBoxRelToGlobalRelCoords(this.attention_curr_pnt);
                this.setNewJoint(globcoords, false);
            }
        }

        this.attention_state = false;
        this.selector_rectangle_is_out = false;
        this.drawbkg();
        this.drawfront();
    }

    drawjoint(joint){
        this.front_ctx.beginPath();
        if(joint.occluded){
            this.front_ctx.fillStyle = OCCLUDEDJOINTCOL;
        }
        else{
            this.front_ctx.fillStyle = VISIBLEJOINTCOL;
        }
        let relcoords = this.fromGlobRelToBBoxRelCoords([joint.px, joint.py]);
        this.front_ctx.arc(relcoords[0] * this.front_canvas.width, relcoords[1] * this.front_canvas.height,
            this.jointRadius, 0, 2 * Math.PI);
        this.front_ctx.fill();
    }

    drawfront(){
        this.front_ctx.clearRect(0, 0, this.front_canvas.width, this.front_canvas.height);
        for(let i = 0; i < this.joints.length; i++){
            this.drawjoint(this.joints[i]);
        }
        if(this.shouldConsiderZoomSelection()){
            this.front_ctx.beginPath();
            this.front_ctx.strokeStyle = SELECTIONRECTCOL;
            this.front_ctx.setLineDash([5]);
            this.front_ctx.lineWidth = 2;
            this.front_ctx.strokeRect(this.attention_start_pnt[0] * this.front_canvas.width,
                this.attention_start_pnt[1] * this.front_canvas.height,
                (this.attention_curr_pnt[0]-this.attention_start_pnt[0])*this.front_canvas.width,
                (this.attention_curr_pnt[1]-this.attention_start_pnt[1])*this.front_canvas.height);
            this.front_ctx.stroke();
        }
    }

    drawbkg(){
        let clipstartx = this.current_bb_x[0] * this.bkgimg.width;
        let clipstarty = this.current_bb_y[0] * this.bkgimg.height;
        let clipwidth = (this.current_bb_x[1] - this.current_bb_x[0]) * this.bkgimg.width;
        let clipheight = (this.current_bb_y[1] - this.current_bb_y[0]) * this.bkgimg.height;
        this.bkg_ctx.drawImage(this.bkgimg,
            clipstartx, clipstarty,
            clipwidth, clipheight,
            0, 0, this.bkg_canvas.width, this.bkg_canvas.height);
    }

    shouldConsiderZoomSelection(){
        if(!this.attention_state)
            return false;
        if(Math.abs(this.attention_start_pnt[0] - this.attention_curr_pnt[0]) > SELRECTTHRESH)
            return true;
        if(Math.abs(this.attention_start_pnt[1] - this.attention_curr_pnt[1]) > SELRECTTHRESH)
            return true;
        return false;
    }

    setNewJoint(coords, occludedFlag){
        if(this.joints.length < target_joints.length) {
            this.joints.push(new Point(coords[0], coords[1], occludedFlag));
            helperhand.next();
        }
    }

    popLastJoint(){
        this.joints.pop();
        helperhand.prev();
    }

    undo(){
        this.popLastJoint();
        this.drawfront();
    }

    fromBBoxRelToGlobalRelCoords(coords) {
        let globx = this.current_bb_x[0] + coords[0] * (this.current_bb_x[1] - this.current_bb_x[0]);
        let globy = this.current_bb_y[0] + coords[1] * (this.current_bb_y[1] - this.current_bb_y[0]);
        return [globx, globy];
    }

    fromGlobRelToBBoxRelCoords(coords) {
        let bbx = (coords[0]-this.current_bb_x[0]) / (this.current_bb_x[1] - this.current_bb_x[0]);
        let bby = (coords[1]-this.current_bb_y[0]) / (this.current_bb_y[1] - this.current_bb_y[0]);
        return [bbx, bby];
    }

}

function getRelativeCoords(event){
    let tg = event.target;
    let dim = tg.getBoundingClientRect();
    let x = event.clientX - dim.left;
    let y = event.clientY - dim.top;

    x = x / dim.width;
    y = y / dim.height;
    return [x, y];
}