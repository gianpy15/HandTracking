var junctions;
var l_canvas = document.getElementById("left_c_points");
var r_canvas = document.getElementById("right_c_points");
var undo_b = document.getElementById("undo_b");
var submit_b = document.getElementById("submit_b");
var new_sample_b = document.getElementById("new_sample_b");

l_canvas.addEventListener('click', e => on_click_events(e, 'l'), false);
l_canvas.addEventListener('contextmenu', e => on_click_events(e, 'r'), false);
l_canvas.oncontextmenu = function() { return false; };
undo_b.addEventListener('click', () => junctions.undo(), false);
submit_b.addEventListener('click', e => on_commit_event(e, true), true);
new_sample_b.addEventListener('click', () => on_commit_event(false), false);

window.onload = function () {
    loadImages();
    junctions = new Points(l_canvas, r_canvas, target_joints);
};

function loadImages() {
    try{
        var l_canvas = document.getElementById("left_c");
        var r_canvas = document.getElementById("right_c");
        var l_ctx = l_canvas.getContext("2d");
        var r_ctx = r_canvas.getContext("2d");

        var left_img = new Image();
        var right_img = new Image();

        right_img.onload = () => r_ctx.drawImage(right_img, 0, 0, r_canvas.width, r_canvas.height);
        left_img.onload = () => l_ctx.drawImage(left_img, 0, 0, l_canvas.width, l_canvas.height);

        left_img.src = target_img_url;
        right_img.src = sample_img_url;
    } catch(error) {
		alert(error)
	}
}
