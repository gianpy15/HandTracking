var helperhand;
var pinpointer;
var l_canvas = document.getElementById("left_c_points");
var r_canvas = document.getElementById("right_c_points");
var l_canvas_bkg = document.getElementById("left_c");
var r_canvas_bkg = document.getElementById("right_c");
var undo_b = document.getElementById("undo_b");
var submit_b = document.getElementById("submit_b");
var new_sample_b = document.getElementById("new_sample_b");

l_canvas.oncontextmenu = function() { return false; };
undo_b.addEventListener('click', () => pinpointer.undo(), false);
submit_b.addEventListener('click', () => submit_and_exit(), true);
new_sample_b.addEventListener('click', () => submit_and_next_frame(), false);
window.addEventListener("beforeunload", () => leavePage());

window.onload = function () {
    loadSampleImage();
    pinpointer = new Pinpointer(l_canvas, l_canvas_bkg, target_img_url);
    helperhand = new HelperHandManager(r_canvas, target_joints);
};

function loadSampleImage() {
    try{
        var r_ctx = r_canvas_bkg.getContext("2d");

        var right_img = new Image();

        right_img.onload = () => r_ctx.drawImage(right_img, 0, 0, r_canvas.width, r_canvas.height);

        right_img.src = sample_img_url;
    } catch(error) {
		alert(error)
	}
}
