function submit_and_exit() {
    if(pinpointer.joints.length === target_joints.length) {
        let data = get_data(pinpointer.joints);

        let params = "labels="+data.toString()+"&framename="+target_img_url+"&nick="+nickname;
        function action(resp){
             let nick_append = (nickname != "") ? "?nick="+nickname : "";
             if (resp != "OK") {
                 console.log(resp);
             }
             location.href = "thanks.html"+nick_append;
        }
        sendPost("submit_labels.php", params, action);
    }
    else
        alert("Please fill all the points");
}

function submit_and_next_frame(){
    if(pinpointer.joints.length === target_joints.length) {
        let data = get_data(pinpointer.joints);

        let params = "labels="+data.toString()+"&framename="+target_img_url+"&nick="+nickname;
        function loadnew(resp){
            target_img_url = resp;

            helperhand.reset();
            pinpointer.resetJoints();
            console.log(resp);
            pinpointer.setBkgUrl(resp);
        }
        sendPost("next_frame.php", '', loadnew);
        sendPost("submit_labels.php", params, (r) => console.log(r));
    }
    else
        alert("Please fill all the points");
}


function sendPost(destination, params='', responseAction=(resp) => null){
    let request = new XMLHttpRequest();
    request.open('POST', destination, true);
    request.onreadystatechange = function () {
        if(request.readyState === XMLHttpRequest.DONE && request.status === 200){
            responseAction(request.responseText);
        }
    };
    request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    request.send(params);
}

function leavePage(){
    alert("leaving");
    sendPost("leaving.php", "framename="+target_img_url, (r) => alert(r));
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
