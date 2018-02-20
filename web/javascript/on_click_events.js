class Point {
   constructor(px, py, occluded){
      this.px = px;
      this.py = py;
      this.occluded = occluded;
   }
}

class Points {
   constructor(canvas/* Add a file where save the points*/) {
      this.N_POINTS = 3;
      // Pixel width
      this.PW = 10;
      this.selected_points = [];

      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.ctx.fillStyle="#FF0000";
   }

   add_point(px, py, occluded) {
      occluded ? this.ctx.fillStyle="#4CAF50" : this.ctx.fillStyle="#F44336";
      if (this.selected_points.length < this.N_POINTS) {
         this.selected_points.push(new Point(px, py, occluded));
         this.ctx.fillRect(px * this.canvas.width,
            py * this.canvas.height, this.PW, this.PW);
      }
      this.print_on_console();
   }

   undo() {
      if(this.selected_points.length > 0) {
         var point = this.selected_points.pop();
         this.ctx.clearRect(point.px * this.canvas.width,
            point.py * this.canvas.height, this.PW, this.PW)
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


var canvas = document.getElementById("left_c_points");
var undo_b = document.getElementById("undo_b");
var submit_b = document.getElementById("submit_b");
// TODO: completare questo bottone
var new_sample_b = document.getElementById("new_sample_b");
let junctions = new Points(canvas);


canvas.addEventListener('click', e => on_click_events(e, 'l'), false);
canvas.addEventListener('contextmenu', e => on_click_events(e, 'r'), false);
undo_b.addEventListener('click', e => junctions.undo(), false);
submit_b.addEventListener('click', e => on_commit_event(e, true), false);
new_sample_b.addEventListener('click', e => on_commit_event(e, false), false);



function on_click_events(e, type) {
   var ctx = canvas.getContext("2d")
   var x = e.target;
   var dim = x.getBoundingClientRect();
   var x = e.clientX - dim.left;
   var y = e.clientY - dim.top;

   x = x / canvas.width;
   y = y / canvas.height;

   if (type == 'l')
      junctions.add_point(x, y, true);
   if (type == 'r')
      junctions.add_point(x, y, false);
}

function on_commit_event(e, exit) {
   if(junctions.selected_points.length == junctions.N_POINTS) {
      var data = get_data(junctions);
      console.log(data.toString());
      // Create the server request
      // var httpc = new XMLHttpRequest();
      var url = "test.php";

      /*
      httpc.open("POST", url, true);
      httpc.setRequestHeader("Content-Type", "application/data");
      httpc.setRequestHeader("labels", data.toString());
      httpc.setRequestHeader("new_image", !exit);

      httpc.onreadystatechange = function() { //Call a function when the state changes.
         if(httpc.readyState == 4 && httpc.status == 200) {// complete and no errors
            if (exit) {
               location.href='thanks.html';
            } else {
               //TODO: Manage the new image
               drawImage(/* New image / document.getElementById("left_c").getContext("2d"));
               junctions.reset();
            }
         } else {
            if (httpc.status != 200) alert("Error sending data to server");
         }
      }

    httpc.send();*/

    var data = $.post(url, {labels: data.toString()})
      .done(data => console.log("Data loaded: " + data)
      .success((response) => on_response(response, exit)));

   }
   else
      alert("Please fill all the points");
}

function get_data(points) {
   var ret = [];
   for (var i = 0; i < points.selected_points.length; i++) {
      ret.push(points.selected_points[i].px);
      ret.push(points.selected_points[i].py);
      ret.push(points.selected_points[i].occluded);
   }
   return ret;
}

function on_response(response, exit_condition) {
   if (exit) {
      location.href='thanks.html';
   } else {
      //TODO: Manage the new image
      drawImage(/* New image */ document.getElementById("left_c").getContext("2d"));
      junctions.reset();
   }
}


function drawImage(/*image_data,*/ context) {
   try {
	   var other_url = 'https://cdn.sstatic.net/stackexchange/img/logos/so/so-icon.png'
	   var left_img = new Image();
      left_img.src = other_url;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.putImageData(left_img, 0, 0, canvas.width, canvas.height);
    } catch(error) {
		alert(error)
	}
}
