function on_commit_event(exit) {
    if(pinpointer.joints.length === target_joints.length) {
        let data = get_data(pinpointer.joints);
        console.log(data.toString());
        console.log(nickname);

        let labels_resp = new XMLHttpRequest();
        labels_resp.open('POST', "submit_labels.php", true);
        labels_resp.onreadystatechange = function () {
            if(labels_resp.readyState === XMLHttpRequest.DONE && labels_resp.status === 200){
                let nick_append = (nickname != "") ? "?nick="+nickname : "";
                if (labels_resp.responseText != "OK") {
                    alert(labels_resp.responseText);
                }
                location.href = exit ? "thanks.html"+nick_append: "prepare_data.php"+nick_append;
            }
        };
        labels_resp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        params = "labels="+data.toString()+"&framename="+target_img_url+"&nick="+nickname;
        labels_resp.send(params);
    }
    else
        alert("Please fill all the points");
}

function get_data(points) {
   let ret = [];
   for (let i = 0; i < points.length; i++) {
      ret.push(points[i].px);
      ret.push(points[i].py);
      ret.push(points[i].occluded);
   }
   return ret;
}
