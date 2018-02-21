function on_click_events(e, type) {
    var x = e.target;
    var dim = x.getBoundingClientRect();
    var x = e.clientX - dim.left;
    var y = e.clientY - dim.top;

    x = x / l_canvas.width;
    y = y / l_canvas.height;

    if (type === 'l')
        junctions.add_point(x, y, true);
    if (type === 'r')
        junctions.add_point(x, y, false);
}

function on_commit_event(exit) {
    if(junctions.selected_points.length === junctions.N_POINTS) {
        var data = get_data(junctions);
        console.log(data.toString());
        console.log(nickname);

        let labels_resp = new XMLHttpRequest();
        labels_resp.open('POST', "submit_labels.php", true);
        labels_resp.onreadystatechange = function () {
            if(labels_resp.readyState === XMLHttpRequest.DONE && labels_resp.status === 200){
                var nick_append = (nickname != "") ? "?nick="+nickname : "";
                if (labels_resp.responseText != "OK") {
                    alert(labels_resp.responseText);
                }
                location.href = exit ? "prepare_data.php"+nick_append : "thanks.html"+nick_append;
            }
        };
        labels_resp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        params = "labels="+data.toString()+"&framename="+target_img_url+"&nick="+nickname;
        // labels_resp.setRequestHeader("labels", data.toString());
        // labels_resp.setRequestHeader("framename", target_img_url);
        // labels_resp.setRequestHeader("nick", nickname);
        labels_resp.send(params);
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
